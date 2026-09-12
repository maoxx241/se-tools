// Source-backed, conditional capacity cases. See analysis/KV-EP-SPECS.md.
import { integer, groupAllGather } from './collectives.mjs';
import { readModelConfig } from './model-catalog.mjs';
import { communicationModels } from './communication-requirements.mjs';
import { modelSpec } from '../scripts/model-analysis-specs.mjs';
import { v41Topology } from './deepseek-v41.mjs';

export const KV_EP_DEFAULTS = Object.freeze({ promptTokens: 262144, requestsPerDP: 16,
  prefillTP: 8, blockSize: 128, speculativeSlots: 7,
  prefillTokensPerDP: 16384, decodeTokensPerDP: 128,
  dispatchBytes: 2, dispatchScaleBytes: 0, combineBytes: 2, countBytes: 4 });
const ceil = (x, d) => (x + d - 1n) / d;
const align = (x, d) => ceil(x, d) * d;

export function balancedEP({ ep, tokensPerRank, layers, hidden, topK, experts,
  dispatchBytes = 2, dispatchScaleBytes = 0, combineBytes = 2, countBytes = 4 }) {
  const P = integer(ep, 'ep', 1n), T = integer(tokensPerRank, 'tokensPerRank');
  const L = integer(layers, 'layers'), H = integer(hidden, 'hidden', 1n);
  const K = integer(topK, 'topK', 1n), E = integer(experts, 'experts', 1n);
  const D = integer(dispatchBytes, 'dispatchBytes', 1n), S = integer(dispatchScaleBytes, 'dispatchScaleBytes');
  const C = integer(combineBytes, 'combineBytes', 1n), Q = integer(countBytes, 'countBytes', 1n);
  if (!((D === 2n && S === 0n) || (D === 1n && S === 4n)) || C !== 2n)
    throw new Error('This A3 model supports BF16/BF16 or INT8+FP32 scale/BF16 only');
  if (K > E) throw new Error('topK exceeds expert count');
  const routes = L * T * K * (P - 1n); // sum over i != j; uniform-rank routing expectation
  const physicalExperts = ceil(E, P) * P;
  const dispatchPayload = routes * (H * D + S), combinePayload = routes * H * C;
  // Explicit ideal Ring model for the dispatcher's per-expert histogram AllGather.
  const histogram = T === 0n ? 0n : L * groupAllGather(physicalExperts * Q, P);
  // A3 FullMesh branch: 480 data bytes + 32 flag bytes in each 512-byte block.
  const dispatchRecord = ceil(align(align(H * D, 32n) + S, 32n) + 32n, 480n) * 512n;
  const dispatchCounts = T === 0n ? 0n : L * physicalExperts * (P - 1n) * 32n;
  const combineFlags = routes * 32n;
  return { routes, physicalExperts, redundantExpertsRequired: physicalExperts - E,
    placementStatus: E % P === 0n ? 'even-expert-count; runtime unverified' : 'conditional: requires expert remap/EPLB; raw config incompatible',
    alltoallv: { dispatchPayload, combinePayload, histogramRingBytes: histogram,
      modeledBytes: dispatchPayload + combinePayload + histogram },
    mc2: { branch: 'A3 dispatch FullMesh + BF16 CombineV2', dispatchRecordBytes: dispatchRecord,
      dispatchPackedBytes: routes * dispatchRecord, dispatchCountBytes: dispatchCounts,
      combinePayload, combineFlagBytes: combineFlags,
      modeledRemoteWriteBytes: routes * dispatchRecord + dispatchCounts + combinePayload + combineFlags,
      // Separate kernel branch; a window stride is NOT a transmitted byte count.
      ordinaryDispatchRecordBytes: align(align(H * D, 32n) + S, 32n) + 12n,
      physicalLinkBytes: null, pollingReadBytes: null },
  };
}

export function kvComponents(model, config, options = {}) {
  const o = { ...KV_EP_DEFAULTS, ...options };
  const S = integer(o.promptTokens, 'promptTokens', 2n), TP = integer(o.prefillTP, 'prefillTP', 1n);
  const B = integer(o.blockSize, 'blockSize', 1n), spec = integer(o.speculativeSlots, 'speculativeSlots');
  if (TP !== 8n) throw new Error('This source-backed case fixes Prefill TP8; add a separately traced topology to change it');
  if (![32n, 64n, 128n].includes(B)) throw new Error('blockSize must be 32, 64, or 128 for these DSV4 cache layouts');
  const c = config.text_config || config;
  const ds = model.profile === 'deepseek_v4', glm = model.profile === 'glm53', v41 = model.profile === 'deepseek_v41';
  const DTP = ds || glm || v41 ? 1n : 8n;
  // V4.1 has no supported connector: snapshot after a full Prefill, without
  // inventing an S-1 recomputation policy for its request-private ring state.
  const N = glm || v41 ? S : S - 1n;
  const parts = [];
  function add(component, layers, rowBytes, generatedRows, retainedRows, pullRows, pCopies = TP, dCopies = DTP, kind = 'kv') {
    const r = { component, kind, layers: BigInt(layers), rowBytes: BigInt(rowBytes), generatedRows,
      retainedRows, pullRows, prefillCopies: pCopies, decodeCopies: dCopies };
    r.generatedBytes = r.layers * r.rowBytes * generatedRows * pCopies;
    r.retainedBytes = r.layers * r.rowBytes * retainedRows * pCopies;
    const transferBytes = r.layers * r.rowBytes * pullRows * dCopies;
    r.pullBytes = v41 ? null : transferBytes;
    if (v41) {
      r.plannedPullRows = pullRows; r.pullRows = null;
      r.plannedPullBytes = transferBytes;
    }
    parts.push(r);
  }
  // Scheduler keeps ceil(window / block)+1 tail blocks and excludes placeholders.
  // This conservative block count is exact for the default N=256K-1 case.
  const tail = (window, block) => (ceil(window, block) + 1n < ceil(N, block) ? ceil(window, block) + 1n : ceil(N, block)) * block;
  if (ds) {
    const ratios = c.compress_ratios.slice(0, c.num_hidden_layers);
    const c4 = ratios.filter(x => x === 4).length, c128 = ratios.filter(x => x === 128).length;
    if (c4 + c128 !== c.num_hidden_layers) throw new Error('Unmodeled DeepSeek compression pattern');
    const H = BigInt(c.head_dim), J = BigInt(c.index_head_dim), W = BigInt(c.sliding_window);
    add('SWA KV / BF16', c.num_hidden_layers, H * 2n, N, N < W ? N : W, tail(W, B));
    for (const [ratio, layers] of [[4n, c4], [128n, c128]])
      add(`C${ratio} KV / BF16`, layers, H * 2n, N / ratio, N / ratio, ceil(ceil(N, ratio), B) * B);
    add('C4 Indexer / INT8 + FP16 scale', c4, J + 2n, N / 4n, N / 4n, ceil(ceil(N, 4n), B) * B, TP, DTP, 'index');
    add('C4 compressor state / FP32', c4, 4n * H * 4n, N < 8n ? N : 8n, N < 8n ? N : 8n, tail(8n, B / 16n), TP, DTP, 'state');
    add('C128 compressor state / FP32', c128, 2n * H * 4n, N < 128n ? N : 128n, N < 128n ? N : 128n, tail(128n, B / 4n), TP, DTP, 'state');
    add('C4 indexer state / FP32', c4, 4n * J * 4n, N < 8n ? N : 8n, N < 8n ? N : 8n, tail(8n, B / 16n), TP, DTP, 'state');
  } else if (v41) {
    const topology = v41Topology(config), H = BigInt(c.head_dim), J = BigInt(c.index_head_dim), W = BigInt(c.sliding_window);
    const c2 = topology.layers.filter(x => x.ownsKV && x.ratio === 2).length;
    add('SWA KV / BF16', c.num_hidden_layers, H * 2n, N, N < W ? N : W, tail(W, B));
    for (const ratio of [2n, 1n]) {
      const count = topology.layers.filter(x => x.ownsKV && x.ratio === Number(ratio)).length;
      // V4.1 uses original-token block_size, physical rows = B / ratio.
      // Only completed pairs have long KV / index K; trailing residual stays in ring.
      const physicalBlock = B / ratio, generated = N / ratio;
      const transferRows = ceil(generated, physicalBlock) * physicalBlock;
      add(`C${ratio} shared KV / BF16`, count, H * 2n, generated, generated, transferRows);
      add(`C${ratio} shared Indexer / INT8 + FP16 scale`, count, J + 2n, generated, generated, transferRows, TP, DTP, 'index');
    }
    const validRing = N < 32n ? N : 32n;
    add('C2 circular ring / FP32 KV+score', c2, 2n * H * 4n, validRing, validRing, 32n, TP, DTP, 'state');
  } else if (glm) {
    const idx = Array.from({ length: c.num_hidden_layers }, (_, i) => i)
      .filter(i => Math.max(i - c.index_skip_topk_offset + 1, 0) % c.index_topk_freq === 0).length;
    add('MLA / BF16', c.num_hidden_layers, BigInt(c.kv_lora_rank + c.qk_rope_head_dim) * 2n, N, N, ceil(N, B) * B);
    add('Indexer / BF16', idx, BigInt(c.index_head_dim) * 2n, N, N, ceil(N, B) * B, TP, DTP, 'index');
  } else {
    let layers, convWidth, recurrentElements, kernel;
    if (model.profile === 'kimi') {
      const k = c.linear_attn_config;
      layers = k.kda_layers.length; kernel = BigInt(k.short_conv_kernel_size);
      convWidth = BigInt(3 * k.num_heads * k.head_dim);
      recurrentElements = BigInt(k.num_heads * k.head_dim * k.head_dim);
      add('MLA replicated / BF16', c.num_hidden_layers - layers, BigInt(c.kv_lora_rank + c.qk_rope_head_dim) * 2n,
        N, N, ceil(N, B) * B);
    } else if (model.profile === 'qwen38') {
      layers = c.layer_types.filter(x => x === 'linear_attention').length; kernel = BigInt(c.linear_conv_kernel_dim);
      convWidth = BigInt(2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim);
      recurrentElements = BigInt(c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim);
      const localHeads = BigInt(Math.max(1, c.num_key_value_heads / Number(TP)));
      add('GQA K+V / BF16 (KV heads replicated)', c.num_hidden_layers - layers, 2n * localHeads * BigInt(c.head_dim) * 2n,
        N, N, ceil(N, B) * B);
    } else throw new Error(`Unsupported KV profile: ${model.profile}`);
    const validConv = N < kernel - 1n ? N : kernel - 1n;
    add('Conv state / BF16', layers, convWidth / TP * 2n, validConv, validConv, kernel - 1n + spec, TP, DTP, 'state');
    add('Recurrent state / FP32', layers, recurrentElements / TP * 4n, 1n, 1n, 1n, TP, DTP, 'state');
  }
  return { prefillTokens: N, prefillTP: TP, decodeTP: DTP, components: parts,
    generatedBytes: parts.reduce((s, p) => s + p.generatedBytes, 0n),
    retainedBytes: parts.reduce((s, p) => s + p.retainedBytes, 0n),
    pullBytes: v41 ? null : parts.reduce((s, p) => s + p.pullBytes, 0n),
    ...(v41 ? { plannedPullBytes: parts.reduce((s, p) => s + p.plannedPullBytes, 0n),
      transferStatus: 'unsupported-in-pinned-main; planned tensor requirement only',
      handoff: 'snapshot after S prefill tokens; D TP1; unique long/index planes, SWA tail-block bound, full 32-row rings; no prefix hit or draft',
      transferExclusions: ['allocator/shared-slot padding', 'protocol and request metadata', 'Engram history', 'DSpark/draft state'],
    } : {}),
    basis: 'per request, all P or D TP ranks; generated KV rows plus one final auxiliary state; no cache hits; backbone only',
    physicalPrefillHBMWriteBytes: null, physicalPDLinkBytes: null };
}

export async function loadKvEpSpecs(options = {}) {
  const o = { ...KV_EP_DEFAULTS, ...options }, rows = [];
  const R = integer(o.requestsPerDP, 'requestsPerDP', 1n);
  const selected = [...communicationModels.slice(0, 5), { name: 'DS V4.1 Flash', slug: 'deepseek-v4.1-flash', profile: 'deepseek_v41' }];
  for (const model of selected) {
    const config = await readModelConfig(model.slug);
    const c = config.text_config || config;
    const facts = modelSpec(model, config).facts;
    // Current case counts target backbone calls; historical GLM added one MTP layer.
    if (model.profile === 'glm53') { facts.layers = c.num_hidden_layers; facts.moeLayers = c.num_hidden_layers - c.mlp_layer_types.filter(x => x === 'dense').length; }
    const cache = kvComponents(model, config, o);
    const transferTotals = copies => cache.pullBytes === null
      ? { pullBytes: null, plannedPullBytes: cache.plannedPullBytes * copies }
      : { pullBytes: cache.pullBytes * copies };
    for (const ep of [32, 256]) {
      const P = BigInt(ep), pDP = P / cache.prefillTP, dDP = P / cache.decodeTP;
      const common = { ep, layers: facts.moeLayers, hidden: facts.H, topK: facts.topK, experts: facts.experts,
        dispatchBytes: o.dispatchBytes, dispatchScaleBytes: o.dispatchScaleBytes, combineBytes: o.combineBytes, countBytes: o.countBytes };
      const pTokens = ceil(integer(o.prefillTokensPerDP, 'prefillTokensPerDP'), cache.prefillTP);
      const dTokens = ceil(integer(o.decodeTokensPerDP, 'decodeTokensPerDP'), cache.decodeTP);
      const pEP = balancedEP({ ...common, tokensPerRank: pTokens }), dEP = balancedEP({ ...common, tokensPerRank: dTokens });
      rows.push({ model: model.name, slug: model.slug, profile: model.profile, ep, facts, options: o,
        topology: { prefillTP: cache.prefillTP, prefillDP: pDP, decodeTP: cache.decodeTP, decodeDP: dDP },
        stepWorkload: { prefillTokensPerRank: pTokens, decodeTokensPerRank: dTokens,
          prefillTokensPerEPGroup: pTokens * P, decodeTokensPerEPGroup: dTokens * P,
          independentMoEDPAllGather: false,
          scope: 'All2All/MC2 token shards already span all DP ranks; do not add the alternative AllGather backend' },
        placementStatus: pEP.placementStatus, redundantExpertsRequired: pEP.redundantExpertsRequired,
        cache, batch: { requests: R, generatedBytes: cache.generatedBytes * R, retainedBytes: cache.retainedBytes * R, ...transferTotals(R) },
        wholePrefillGroup: { requests: R * pDP, generatedBytes: cache.generatedBytes * R * pDP, ...transferTotals(R * pDP) },
        prefillEP: pEP.alltoallv, decodeEP: dEP.mc2,
        epPerRank: { prefillMeanBytes: { numerator: pEP.alltoallv.modeledBytes, denominator: P },
          decodeMeanBytes: { numerator: dEP.mc2.modeledRemoteWriteBytes, denominator: P } },
      });
    }
  }
  return rows;
}
