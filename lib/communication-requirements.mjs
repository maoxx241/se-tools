import { modelSpec } from "../scripts/model-analysis-specs.mjs";
import { readModelConfig } from "./model-catalog.mjs";
import { integer, groupAllReduce, groupAllGather, groupReduceScatter, groupEqualAllToAll } from "./collectives.mjs";

// Reproduction of the final 256K case. Model call placement and cache layouts
// remain session assumptions; HCCL only supports the collective-count layer.
const TP = 8n;
const EP = 32n;
const PREFILL_TOKENS_PER_STEP = 16_384n;
const DECODE_TOKENS_PER_STEP = 128n; // 16 requests * (7 speculative tokens + 1)
const INPUT_TOKENS_PER_REQUEST = 262_144n;
const REQUESTS_PER_DP = 16n;
const ACTIVATION_BYTES = 2n;
const INDEX_BYTES = 4n;
const CANDIDATE_K = 256n;

export function calculateStepTraffic(
  facts,
  phaseTokens,
  samplingRows,
  { tp, ep, spEnabled, dsa, sharedExpertTpEnabled },
) {
  const T = integer(phaseTokens, "phaseTokens");
  tp = integer(tp, "tp", 1n);
  ep = integer(ep, "ep", 1n);
  samplingRows = integer(samplingRows, "samplingRows");
  if (dsa && !spEnabled) throw new Error("This DSA case requires SP token sharding");
  const H = BigInt(facts.H);
  const attentionLayers = BigInt(facts.attentionLayers);
  const denseLayers = BigInt(facts.denseLayers || 0);
  const moeLayers = BigInt(facts.moeLayers || 0);
  const localTokens = spEnabled ? (T + tp - 1n) / tp : T;
  const paddedTokens = spEnabled ? localTokens * tp : T;

  const embedding = groupAllReduce(T * H * ACTIVATION_BYTES, tp);
  const lmHead = groupAllGather(samplingRows * CANDIDATE_K * (ACTIVATION_BYTES + INDEX_BYTES), tp);

  const finalModelOutput = spEnabled
    ? groupAllGather(localTokens * H * ACTIVATION_BYTES, tp)
    : 0n;
  const denseMlp = denseLayers * groupAllReduce(localTokens * H * ACTIVATION_BYTES, tp);
  const tpBoundary = embedding + lmHead + (spEnabled ? 0n : denseMlp);

  let attention = 0n;
  let sp = finalModelOutput + (spEnabled ? denseMlp : 0n);
  let dsaCp = 0n;
  if (dsa) {
    const qKvInput = attentionLayers * groupAllGather(localTokens * H * ACTIVATION_BYTES, tp);
    const restoredHeadLayout = attentionLayers * groupEqualAllToAll(
      localTokens * BigInt(facts.heads) * BigInt(facts.headDim) * ACTIVATION_BYTES,
      tp,
    );
    const indexer = BigInt(facts.indexerLayers || 0) * groupAllGather(
      localTokens * BigInt(facts.indexTopK || 0) * INDEX_BYTES,
      tp,
    );
    dsaCp = qKvInput + restoredHeadLayout + indexer;
    attention = dsaCp;
  } else {
    const beforeAttention = attentionLayers * groupAllGather(localTokens * H * ACTIVATION_BYTES, tp);
    const afterOProj = attentionLayers * groupReduceScatter(paddedTokens * H * ACTIVATION_BYTES, tp);
    attention = spEnabled ? beforeAttention + afterOProj
      : attentionLayers * groupAllReduce(T * H * ACTIVATION_BYTES, tp);
    if (spEnabled) sp += attention;
  }

  let epAllToAll = 0n;
  let sharedExpertTp = 0n;
  if (moeLayers > 0n) {
    // Uniform expert routing expectation: (EP-1)/EP of assignments cross ranks.
    // Multiplying by the whole EP group cancels the /EP route fraction.
    const oneDirection = localTokens
      * BigInt(facts.topK)
      * (ep - 1n)
      * H
      * ACTIVATION_BYTES;
    epAllToAll = moeLayers * 2n * oneDirection; // dispatch + combine
    if (sharedExpertTpEnabled) {
      const sharedInput = groupAllGather(localTokens * H * ACTIVATION_BYTES, tp);
      const sharedOutput = groupReduceScatter(paddedTokens * H * ACTIVATION_BYTES, tp);
      sharedExpertTp = moeLayers * (spEnabled ? sharedInput + sharedOutput
        : groupAllReduce(T * H * ACTIVATION_BYTES, tp));
    }
  }

  return { tpBoundary, sp, dsaCp, attention, epAllToAll, sharedExpertTp };
}

// Disjoint Decode TP/SP event ledger for the archived SP service path.
// All values are average SENT bytes per rank, not per TP group. Attention is
// already part of SP in calculateStepTraffic; never add that alias twice.
export function decodeTpTraffic(facts, tokens, tp, { sharedExpertTpEnabled = false } = {}) {
  const T = integer(tokens, 'tokens'), P = integer(tp, 'tp', 1n);
  const H = BigInt(facts.H), local = (T + P - 1n) / P;
  const perRank = bytes => Number(bytes) / Number(P);
  const pair = groupAllGather(local * H * 2n, P) + groupReduceScatter(local * P * H * 2n, P);
  const components = {
    embedding: perRank(groupAllReduce(T * H * 2n, P)),
    attention: perRank(BigInt(facts.attentionLayers) * pair),
    denseMlp: perRank(BigInt(facts.denseLayers || 0) * groupAllReduce(local * H * 2n, P)),
    sharedExpert: sharedExpertTpEnabled ? perRank(BigInt(facts.moeLayers) * pair) : 0,
    finalHidden: perRank(groupAllGather(local * H * 2n, P)),
    sampling: perRank(groupAllGather(T * CANDIDATE_K * (ACTIVATION_BYTES + INDEX_BYTES), P)),
  };
  return { components, totalBytesPerRank: Object.values(components).reduce((a,b)=>a+b,0) };
}

function qwenDenseFacts(config) {
  const c = config.text_config;
  const full = c.layer_types.filter((value) => value === "full_attention").length;
  return {
    H: c.hidden_size,
    attentionLayers: c.num_hidden_layers,
    fullAttentionLayers: full,
    linearAttentionLayers: c.num_hidden_layers - full,
    denseLayers: c.num_hidden_layers,
    moeLayers: 0,
    heads: c.num_attention_heads,
    headDim: c.head_dim,
    kvHeads: c.num_key_value_heads,
    topK: 0,
    indexTopK: 0,
    indexerLayers: 0,
  };
}

function deepseekPdCache(config, facts) {
  const ratios = config.compress_ratios.slice(0, facts.layers);
  const c4 = BigInt(ratios.filter((value) => value === 4).length);
  const c128 = BigInt(ratios.filter((value) => value === 128).length);
  const layers = BigInt(facts.layers);
  const headDim = BigInt(config.head_dim);
  const indexHeadDim = BigInt(config.index_head_dim);
  const slidingWindow = BigInt(config.sliding_window);
  const tokens = INPUT_TOKENS_PER_REQUEST;

  const compressedKv = (
    c4 * (tokens / 4n)
    + c128 * (tokens / 128n)
  ) * headDim * 2n;
  const swa = layers * slidingWindow * headDim * 2n;

  // The DeepSeek-V4 code keeps compressor state in FP32.
  const c4CompressorState = c4 * (2n * 2n * headDim) * (2n * 4n) * 4n;
  const c128CompressorState = c128 * (2n * headDim) * 128n * 4n;

  // For C4 layers on the non-A5 path, the indexer stores INT8 K plus one FP16 scale.
  const indexer = c4 * (tokens / 4n) * (indexHeadDim + 2n);
  const indexerCompressorState = c4 * (2n * 2n * indexHeadDim) * (2n * 4n) * 4n;

  const attention = REQUESTS_PER_DP * (compressedKv + swa);
  const auxiliary = REQUESTS_PER_DP * (
    c4CompressorState + c128CompressorState + indexer + indexerCompressorState
  );
  return { attention, auxiliary, total: attention + auxiliary };
}

function glmPdCache(config, facts) {
  const tokens = INPUT_TOKENS_PER_REQUEST;
  const mainMla = REQUESTS_PER_DP
    * tokens
    * BigInt(facts.layers)
    * BigInt(config.kv_lora_rank + config.qk_rope_head_dim)
    * 2n;
  // The Decode side is TP1/DCP1, so P→D sends one indexer-cache copy.
  const indexer = REQUESTS_PER_DP
    * tokens
    * BigInt(facts.indexerLayers)
    * BigInt(config.index_head_dim)
    * 2n;
  return { attention: mainMla, auxiliary: indexer, total: mainMla + indexer };
}

function kimiPdCache(config, facts) {
  const c = config.text_config;
  const mla = REQUESTS_PER_DP
    * INPUT_TOKENS_PER_REQUEST
    * BigInt(facts.mlaLayers)
    * BigInt(c.kv_lora_rank + c.qk_rope_head_dim)
    * 2n
    * TP; // MLA cache is replicated across ordinary TP8 ranks.

  const localHeads = BigInt(c.linear_attn_config.num_heads) / TP;
  const headDim = BigInt(c.linear_attn_config.head_dim);
  const convDim = 3n * BigInt(c.linear_attn_config.num_heads) * headDim / TP;
  const stateLen = BigInt(c.linear_attn_config.short_conv_kernel_size - 1 + 7);
  const convBytesPerRank = convDim * stateLen * 2n;
  const recurrentBytesPerRank = localHeads * headDim * headDim * 4n;
  const kdaState = REQUESTS_PER_DP
    * BigInt(facts.kdaLayers)
    * TP
    * (convBytesPerRank + recurrentBytesPerRank);
  return { attention: mla, auxiliary: kdaState, total: mla + kdaState };
}

function qwenPdCache(config, facts) {
  const c = config.text_config || config;
  const localKvHeads = BigInt(Math.max(1, c.num_key_value_heads / Number(TP)));
  const fullAttention = REQUESTS_PER_DP
    * INPUT_TOKENS_PER_REQUEST
    * BigInt(facts.fullAttentionLayers)
    * TP
    * 2n
    * localKvHeads
    * BigInt(c.head_dim)
    * 2n;

  const convDim = (
    2n * BigInt(c.linear_num_key_heads) * BigInt(c.linear_key_head_dim)
    + BigInt(c.linear_num_value_heads) * BigInt(c.linear_value_head_dim)
  ) / TP;
  const stateLen = BigInt(c.linear_conv_kernel_dim - 1 + 7);
  const convBytesPerRank = convDim * stateLen * 2n;
  const recurrentBytesPerRank = (
    BigInt(c.linear_num_value_heads) / TP
  ) * BigInt(c.linear_value_head_dim) * BigInt(c.linear_key_head_dim) * 4n;
  const gdnState = REQUESTS_PER_DP
    * BigInt(facts.linearAttentionLayers)
    * TP
    * (convBytesPerRank + recurrentBytesPerRank);
  return { attention: fullAttention, auxiliary: gdnState, total: fullAttention + gdnState };
}


export const communicationModels = [
  { name: "DS V4 10T", slug: "deepseek-v4-10t", profile: "deepseek_v4" },
  { name: "DS V4 Pro", slug: "deepseek-v4-pro-0813", profile: "deepseek_v4" },
  { name: "GLM5.3", slug: "glm-5.3", profile: "glm53" },
  { name: "KimiK3", slug: "kimi-k3", profile: "kimi" },
  { name: "Qwen3.8 2.4T", slug: "qwen3.8-2.4t-a95b", profile: "qwen38" },
  { name: "Qwen3.8 27B", slug: "qwen3.8-27b", profile: "qwen_dense" },
];

export function calculateRequirement(model, config) {
  const facts = model.profile === "qwen_dense" ? qwenDenseFacts(config) : modelSpec(model, config).facts;
  const dsaModel = ["deepseek_v4", "glm53"].includes(model.profile);
  const sharedExpertTpEnabled = model.profile === "kimi";
  const prefill = calculateStepTraffic(facts, PREFILL_TOKENS_PER_STEP, REQUESTS_PER_DP, {
    tp: TP, ep: EP, spEnabled: true, dsa: dsaModel, sharedExpertTpEnabled,
  });
  const decode = calculateStepTraffic(facts, DECODE_TOKENS_PER_STEP, DECODE_TOKENS_PER_STEP, {
    tp: dsaModel ? 1n : TP, ep: EP, spEnabled: !dsaModel, dsa: false, sharedExpertTpEnabled,
  });
  const pdCache = model.profile === "deepseek_v4" ? deepseekPdCache(config, facts)
    : model.profile === "glm53" ? glmPdCache(config, facts)
    : model.profile === "kimi" ? kimiPdCache(config, facts) : qwenPdCache(config, facts);
  return {
    model: model.name, slug: model.slug, facts, dsaModel, prefill, decode, pdCache,
    topology: { prefill: { tp: 8, dp: 4, ep: 32, sp: true, dsa: dsaModel },
      decode: { tp: dsaModel ? 1 : 8, dp: dsaModel ? 32 : 4, ep: 32, sp: !dsaModel, dsa: false } },
    scope: {
      step: "one named communication group per model step; attention overlaps SP/DSA rows; do not sum all categories",
      ep: "uniform routing expectation; hidden payload only; per-rank balance assumed",
      pdCache: "one source DP batch of 16 requests; 262144 input tokens each; destination copies and allocated state included",
      physicalLinkBytes: null,
    },
  };
}

export async function loadCommunicationRequirements(configRoot, overrides = {}) {
  const results = [];
  for (const model of communicationModels) {
    const config = overrides[model.slug] || await readModelConfig(model.slug, configRoot);
    results.push(calculateRequirement(model, config));
  }
  return results;
}
