// DeepSeek V4.1 Flash: official weight representation, VA runtime geometry.
// Evidence and scope: models/deepseek-v4.1-flash/README.md.
import { integer, groupAllGather, groupAllReduce, groupReduceScatter, groupEqualAllToAll, estimateAllToAllV } from './collectives.mjs';

export const V41_HF_REVISION = 'dba1be0a40aa45a94ad051997016db3960a90277';
export const V41_VA_REVISION = '1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c';

export function v41Topology(config) {
  const c = config.text_config ?? config;
  const alias = (released, legacy) => {
    if (c[released] !== undefined && c[legacy] !== undefined
        && JSON.stringify(c[released]) !== JSON.stringify(c[legacy])) throw new Error(`Conflicting ${released}/${legacy}`);
    return c[released] ?? c[legacy];
  };
  const kv = alias('kv_source_layer_ids', 'kv_source_layers');
  const index = alias('index_source_layer_ids', 'index_source_layers');
  const candidate = alias('candidate_source_layer_id', 'candidate_source_layer');
  const L = Number(integer(c.num_hidden_layers, 'num_hidden_layers', 1n));
  if (!Array.isArray(c.compress_ratios) || c.compress_ratios.length < L) throw new Error('compress_ratios must cover backbone');
  for (const sources of [kv, index]) {
    if (!Array.isArray(sources) || sources.some((x, i) => !Number.isInteger(x) || x < 0 || x >= L || (i && x <= sources[i - 1]))) throw new Error('Sources must be sorted unique layer IDs');
  }
  if (!kv.every(x => index.includes(x)) || !kv.includes(candidate)) throw new Error('Invalid source relationships');
  const layers = c.compress_ratios.slice(0, L).map((ratio, layer) => {
    if (![0, 1, 2].includes(ratio)) throw new Error('V4.1 ratios must be 0/1/2');
    const latest = sources => ratio ? sources.findLast(x => x <= layer) : null;
    const kvSource = latest(kv), indexSource = latest(index);
    if (ratio && (kvSource == null || indexSource == null || c.compress_ratios[kvSource] !== ratio)) throw new Error(`Invalid KV source at layer ${layer}`);
    if (!ratio && (kv.includes(layer) || index.includes(layer))) throw new Error('Local-only layer cannot own long cache');
    return { layer, ratio, kvSource, indexSource, ownsKV: kv.includes(layer), ownsIndex: index.includes(layer), candidateFilter: indexSource != null && indexSource > candidate };
  });
  return { layers, kvSources: kv, indexSources: index, candidateSource: candidate };
}

export function v41WeightSpec(config) {
  const c = config.text_config, v = config.vision_config;
  const topology = v41Topology(config);
  const H = c.hidden_size, D = c.head_dim, L = c.num_hidden_layers, heads = c.num_attention_heads;
  const I = c.moe_intermediate_size, E = c.n_routed_experts, q = c.q_lora_rank, o = c.o_lora_rank, g = c.o_groups;
  const rows = [];
  const add = (module, name, dims, dtype = 'BF16', bytes = 2, count = 1, partition = 'replicated') => rows.push({ module, name, dims, dtype, bytes, count, partition });
  const fp8 = (module, name, out, input, count = 1, partition = 'replicated') => {
    const [br, bc] = config.quantization_config.weight_block_size;
    add(module, `${name}.weight`, [out, input], 'FP8 E4M3', 1, count, partition);
    add(module, `${name}.scale`, [Math.ceil(out / br), Math.ceil(input / bc)], 'E8M0', 1, count, partition);
  };
  const blocks = (prefix, count, experts, draft = false) => {
    const m = draft ? 'DSpark' : 'Attention';
    add(m, `${prefix}.attn.attn_sink`, [heads], 'F32', 4, count, 'dsa_tp');
    fp8(m, `${prefix}.attn.wq_a`, q, H, count);
    fp8(m, `${prefix}.attn.wq_b`, heads * D, q, count, 'dsa_tp');
    fp8(m, `${prefix}.attn.wkv`, D, H, count);
    fp8(m, `${prefix}.attn.wo_a`, g * o, heads * D / g, count, 'tp');
    fp8(m, `${prefix}.attn.wo_b`, H, g * o, count, 'tp');
    for (const [name, dim] of [['attn.q_norm', q], ['attn.kv_norm', D], ['attn_norm', H], ['ffn_norm', H]]) add(m, `${prefix}.${name}.weight`, [dim], 'BF16', 2, count);
    add(draft ? 'DSpark Router' : 'MoE Router', `${prefix}.ffn.gate.weight`, [experts, H], 'BF16', 2, count);
    for (const name of ['bias', 'bias_vl']) add(draft ? 'DSpark Router' : 'MoE Router', `${prefix}.ffn.gate.${name}`, [experts], 'F32', 4, count);
    for (const [name, out, input] of [['w1', I, H], ['w3', I, H], ['w2', H, I]]) {
      add(draft ? 'DSpark Experts' : 'Routed Experts', `${prefix}.ffn.experts.N.${name}.weight`, [out, input], 'FP4 E2M1', .5, count * experts, 'ep_individual');
      add(draft ? 'DSpark Experts' : 'Routed Experts', `${prefix}.ffn.experts.N.${name}.scale`, [out, Math.ceil(input / 32)], 'E8M0', 1, count * experts, 'ep_individual');
      fp8(draft ? 'DSpark Shared' : 'Shared Experts', `${prefix}.ffn.shared_experts.${name}`, name === 'w2' ? H : I * c.n_shared_experts, name === 'w2' ? I * c.n_shared_experts : H, count, 'v41_shared');
    }
    const mix = (2 + c.hc_mult) * c.hc_mult;
    for (const part of ['attn', 'ffn']) {
      add(draft ? 'DSpark HC' : 'HC', `${prefix}.hc_${part}_fn`, [mix, c.hc_mult * H], 'F32', 4, count);
      add(draft ? 'DSpark HC' : 'HC', `${prefix}.hc_${part}_base`, [mix], 'F32', 4, count);
      add(draft ? 'DSpark HC' : 'HC', `${prefix}.hc_${part}_scale`, [3], 'F32', 4, count);
    }
  };
  add('Embedding', 'embed.weight', [c.vocab_size, H], 'BF16', 2, 1, 'embedding');
  add('LM Head', 'head.weight', [c.vocab_size, H], 'BF16', 2, 1, 'lmhead');
  add('Backbone', 'norm.weight', [H]);
  blocks('layers.N', L, E);
  const sources = topology.kvSources.length, c2 = topology.layers.filter(x => x.ownsKV && x.ratio === 2).length;
  add('Compressor', 'layers.N.attn.compressor.wkv.weight', [D, H], 'BF16', 2, sources);
  add('Compressor C2', 'layers.N.attn.compressor.wgate.weight', [D, H], 'BF16', 2, c2);
  add('Compressor', 'layers.N.attn.compressor.norm.weight', [D], 'BF16', 2, sources);
  fp8('Indexer', 'layers.N.attn.indexer.wq_b', c.index_n_heads * c.index_head_dim, q, topology.indexSources.length);
  add('Indexer', 'layers.N.attn.indexer.weights_proj.weight', [c.index_n_heads, H], 'BF16', 2, topology.indexSources.length);
  add('Indexer K', 'layers.N.attn.indexer.wk.weight', [c.index_head_dim, D], 'BF16', 2, sources);
  add('Indexer K', 'layers.N.attn.indexer.k_norm.weight', [c.index_head_dim], 'BF16', 2, sources);
  for (let i = 0; i < c.engram_layer_ids.length; i++) {
    const prefix = `layers.${c.engram_layer_ids[i]}.engram.embed`, n = c.engram_num_embeddings[i];
    add('Engram Table', `${prefix}.weight`, [n, c.engram_head_dim], 'FP8 E4M3', 1, 1, 'node_shard');
    add('Engram Table', `${prefix}.scale`, [n, c.engram_head_dim / 32], 'E8M0', 1, 1, 'node_shard');
  }
  const engrams = c.engram_layer_ids.length, columns = (c.engram_max_ngram_size - 1) * c.engram_n_heads;
  fp8('Engram Gate', 'layers.N.engram.wkv', (c.hc_mult + 1) * H, columns * c.engram_head_dim, engrams);
  for (const name of ['q_weight', 'k_weight']) add('Engram Gate', `layers.N.engram.${name}`, [c.hc_mult, H], 'BF16', 2, engrams);
  blocks('mtp.N', c.num_nextn_predict_layers, c.dspark_n_routed_experts, true);
  fp8('DSpark Input', 'mtp.N.main_proj', H, H * c.dspark_target_layer_ids.length, 1, 'tp');
  for (const name of ['main_norm', 'norm']) add('DSpark Norm', `mtp.N.${name}.weight`, [H]);
  add('DSpark Markov', 'mtp.N.markov_head.embed.weight', [c.vocab_size, c.dspark_markov_rank], 'BF16', 2, 1, 'embedding');
  add('DSpark Markov', 'mtp.N.markov_head.head.weight', [c.vocab_size, c.dspark_markov_rank], 'BF16', 2, 1, 'lmhead');
  add('DSpark Confidence', 'mtp.N.confidence_head.proj.weight', [1, H + c.dspark_markov_rank]);
  for (const name of ['image_start', 'image_end', 'image_newline']) add('Image Tokens', name, [H]);
  const V = v.hidden_size, VI = v.intermediate_size, VL = v.num_hidden_layers;
  add('Vision', 'vision.patch_embed.proj.weight', [V, 3 * v.patch_size ** 2]);
  add('Vision', 'vision.patch_embed.proj.bias', [V]);
  for (const [name, dims] of [['attn.wqkv.weight', [3 * V, V]], ['attn.wqkv.bias', [3 * V]], ['attn.wo.weight', [V, V]], ['attn.wo.bias', [V]], ['mlp.w1.weight', [2 * VI, V]], ['mlp.w2.weight', [V, VI]], ['norm1.weight', [V]], ['norm2.weight', [V]]]) add('Vision', `vision.blocks.N.${name}`, dims, 'BF16', 2, VL);
  add('Vision', 'vision.norm.weight', [V]);
  for (const [name, dims] of [['w1.weight', [H, V * v.downsample_ratio ** 2]], ['w1.bias', [H]], ['w2.weight', [H, H]], ['w2.bias', [H]]]) add('Aligner', `aligner.${name}`, dims);
  return { rows, facts: { H, vocab: c.vocab_size, layers: L, attentionLayers: L, tpAttentionLayers: L, dcpAttentionLayers: 0, denseLayers: 0, moeLayers: L, heads, headDim: D, kvHeads: 1, kvRank: D, ropeDim: c.qk_rope_head_dim, experts: E, topK: c.num_experts_per_tok, indexTopK: c.index_topk, indexerLayers: topology.indexSources.length, engramTables: engrams, engramColumns: columns, engramWidth: c.engram_head_dim, auxLayers: c.dspark_target_layer_ids.length } };
}

export function v41Cache(config, { sequenceTokens = 262144, requests = 16, tp = 1 } = {}) {
  const c = config.text_config, topology = v41Topology(config);
  const T = integer(sequenceTokens, 'sequenceTokens'), R = integer(requests, 'requests'), P = integer(tp, 'tp', 1n);
  const D = BigInt(c.head_dim), I = BigInt(c.index_head_dim);
  let rows = 0n, c2 = 0n;
  for (const layer of topology.layers.filter(x => x.ownsKV)) {
    rows += T / BigInt(layer.ratio);
    if (layer.ratio === 2) c2++;
  }
  const longKV = rows * D * 2n;
  const indexKV = rows * (I + 2n);
  const swa = BigInt(c.num_hidden_layers) * (T < BigInt(c.sliding_window) ? T : BigInt(c.sliding_window)) * D * 2n;
  const state = T ? c2 * 32n * 2n * D * 4n : 0n;
  return { basis: 'one TP replica, logical retained payload; excludes allocator padding/null/free IDs, draft and Engram history',
    sequenceTokens: T, requests: R, tp: P, longRowsPerRequest: rows,
    longKVBytesPerRequest: longKV, indexBytesPerRequest: indexKV, swaBytesPerRequest: swa, ringBytesPerRequest: state,
    batchLogicalBytesPerReplica: (longKV + indexKV + swa + state) * R,
    batchLogicalBytesAcrossTP: (longKV + indexKV + swa + state) * R * P,
    pdTransferBytes: null, pdStatus: 'no validated V4.1 transfer path; cache storage is not measured P-to-D traffic' };
}

// Each count is a lookup ID routed by a TP leader to a node-local owner.
export function v41EngramRouting({ sendCounts, width = 256, compressedInt8Wire = false }) {
  const W = integer(width, 'width', 1n);
  if (compressedInt8Wire && W % 32n) throw new Error('INT8 wire width must be divisible by 32');
  const requests = estimateAllToAllV({ sendCounts, dtypeBytes: 8 });
  const reverse = sendCounts.map((_, dst) => sendCounts.map(row => row[dst]));
  const responseWidth = compressedInt8Wire ? W + W / 32n * 4n : W * 2n;
  const responses = estimateAllToAllV({ sendCounts: reverse, dtypeBytes: responseWidth });
  return { requests, responses, responseBytesPerLookup: responseWidth,
    scope: 'one table, node-local ID and reverse-response exchanges; excludes count AllGather, TP Broadcast and CPU-to-device copies' };
}

export function v41Step(config, { tokens, tp = 8, dp = 4, ep = 32, sp = true, dsaCP = true, nodeRanks = 8, activationBytes = 2, dispatchBytes = 2, combineBytes = 2, engramEnabled = true } = {}) {
  const c = config.text_config;
  const T = integer(tokens, 'tokens'), P = integer(tp, 'tp', 1n), DP = integer(dp, 'dp', 1n), EP = integer(ep, 'ep', 1n), N = integer(nodeRanks, 'nodeRanks', 1n);
  const A = integer(activationBytes, 'activationBytes', 1n), DS = integer(dispatchBytes, 'dispatchBytes', 1n), CB = integer(combineBytes, 'combineBytes', 1n);
  if (EP !== P * DP || EP % N || N % P || BigInt(c.n_routed_experts) % EP || BigInt(c.o_groups) % P) throw new Error('Requires EP=TP*DP, equal complete nodes, node-local TP, divisible experts/O groups');
  if (!sp && P > 1n) throw new Error('Numeric example supports SP or TP=1; non-SP multi-rank MoE finalize needs a selected backend');
  const H = BigInt(c.hidden_size), L = BigInt(c.num_hidden_layers), local = (T + P - 1n) / P, padded = local * P;
  const attention = {
    spInputGroupBytes: sp ? groupAllGather(local * H * A, P) * L : 0n,
    headExchangeGroupBytes: dsaCP ? groupEqualAllToAll(local * BigInt(c.num_attention_heads * c.head_dim) * A, P) * L : 0n,
    outputGroupBytes: (sp ? groupReduceScatter(padded * H * A, P) : groupAllReduce(T * H * A, P)) * L,
    finalHiddenGroupBytes: sp ? groupAllGather(local * H * A, P) : 0n,
    indexTopKGroupBytes: 0n, sharedExpertGroupBytes: 0n,
  };
  const routed = local * BigInt(c.num_experts_per_tok) * (EP - 1n) * L;
  const tables = BigInt(c.engram_layer_ids.length), columns = BigInt((c.engram_max_ngram_size - 1) * c.engram_n_heads);
  // One leader per TP group. Uniform owner expectation; no deduplication.
  const perLeaderLookups = T * tables * columns;
  const nodeLookups = perLeaderLookups * N / P;
  const engram = { basis: 'uniform owner expectation; BF16 wire for all default storage modes',
    metadataRingGroupBytes: engramEnabled ? groupAllGather(tables * (N + 1n) * 8n, N) : 0n,
    idsGroupBytes: engramEnabled ? nodeLookups * 8n * (N - 1n) / N : 0n,
    responseGroupBytes: engramEnabled ? nodeLookups * BigInt(c.engram_head_dim) * 2n * (N - 1n) / N : 0n,
    broadcastTreeGroupBytesPerTP: engramEnabled ? perLeaderLookups * BigInt(c.engram_head_dim) * 2n * (P - 1n) : 0n,
  };
  return { tokens: T, tp: P, dp: DP, ep: EP, nodeRanks: N, sp, dsaCP, attention,
    moeExpected: { dispatchGroupBytes: routed * H * DS, combineGroupBytes: routed * H * CB, basis: 'uniform expert-instance routing, hidden payload only; excludes scales/count metadata; not an MC2 or measured AlltoAllV total' },
    engram, physicalLinkBytes: null,
    excluded: ['embedding/logits communication', 'DSpark forward and auxiliary handoff', 'MoE scale/count metadata, padding policy and backend-specific transport', 'vision preprocessing', 'protocol overhead and idle-DP synchronization'],
    scope: 'attention fields are per TP group; MoE per EP group; Engram per node except Broadcast per TP; do not sum these domains' };
}
