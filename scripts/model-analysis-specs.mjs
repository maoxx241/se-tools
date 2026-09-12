import { v41WeightSpec } from '../lib/deepseek-v41.mjs';

function row(module, name, dims, dtype, bytes, count = 1, partition = "replicated") {
  return { module, name, dims, dtype, bytes, count, partition };
}

function fp8Rows(module, name, outDim, inDim, count, partition, scaleDtype = "F8_E8M0", scaleBytes = 1) {
  return [
    row(module, `${name}.weight`, [outDim, inDim], "FP8 E4M3", 1, count, partition),
    row(module, `${name}.weight_scale`, [Math.ceil(outDim / 128), Math.ceil(inDim / 128)], scaleDtype, scaleBytes, count, partition),
  ];
}

function fp4Rows(module, name, outDim, inDim, count, partition) {
  return [
    row(module, `${name}.weight`, [outDim, inDim], "FP4", 0.5, count, partition),
    row(module, `${name}.weight_scale`, [outDim, Math.ceil(inDim / 32)], "E8M0", 1, count, partition),
  ];
}

function denseMlpRows(module, prefix, hidden, intermediate, count, dtype = "BF16", bytes = 2, partition = "tp") {
  return [
    row(module, `${prefix}.gate_proj.weight`, [intermediate, hidden], dtype, bytes, count, partition),
    row(module, `${prefix}.up_proj.weight`, [intermediate, hidden], dtype, bytes, count, partition),
    row(module, `${prefix}.down_proj.weight`, [hidden, intermediate], dtype, bytes, count, partition),
  ];
}

function kimiSpec(config) {
  const c = config.text_config;
  const v = config.vision_config;
  const H = c.hidden_size;
  const L = c.num_hidden_layers;
  const dense = c.first_k_dense_replace;
  const moe = L - dense;
  const I = c.intermediate_size;
  const MI = c.moe_intermediate_size;
  const E = c.num_experts;
  const sharedI = c.num_shared_experts * MI;
  const kda = c.linear_attn_config.kda_layers.length;
  const mla = L - kda;
  const kdaHeads = c.linear_attn_config.num_heads;
  const kdaDim = c.linear_attn_config.head_dim;
  const D = kdaHeads * kdaDim;
  const qRank = c.q_lora_rank;
  const kvRank = c.kv_lora_rank;
  const qNope = c.qk_nope_head_dim;
  const qRope = c.qk_rope_head_dim;
  const vDim = c.v_head_dim;
  const rows = [
    row("Embedding", "model.embed_tokens.weight", [c.vocab_size, H], "BF16", 2, 1, "embedding"),
    row("LM Head", "lm_head.weight", [c.vocab_size, H], "BF16", 2, 1, "lmhead"),
    row("Backbone", "model.norm.weight", [H], "BF16", 2),
    row("Backbone", "input_layernorm.weight", [H], "BF16", 2, L),
    row("Backbone", "post_attention_layernorm.weight", [H], "BF16", 2, L),
    row("Backbone", "self_attention_res_norm.weight", [H], "BF16", 2, L),
    row("Backbone", "self_attention_res_proj.weight", [1, H], "BF16", 2, L),
    row("Backbone", "mlp_res_norm.weight", [H], "BF16", 2, L),
    row("Backbone", "mlp_res_proj.weight", [1, H], "BF16", 2, L),
    row("Backbone", "output_attn_res_norm.weight", [H], "BF16", 2),
    row("Backbone", "output_attn_res_proj.weight", [1, H], "BF16", 2),
    ...denseMlpRows("Dense MLP", "mlp", H, I, dense),
    row("MoE Router", "block_sparse_moe.gate.weight", [E, H], "BF16", 2, moe),
    row("MoE Router", "block_sparse_moe.gate.e_score_correction_bias", [E], "F32", 4, moe),
    row("Latent MoE", "routed_expert_up_proj.weight", [H, H / 2], "BF16", 2, moe),
    row("Latent MoE", "routed_expert_norm.weight", [H / 2], "BF16", 2, moe),
    row("Latent MoE", "routed_expert_down_proj.weight", [H / 2, H], "BF16", 2, moe),
    ...fp4Rows("Routed Experts", "experts.w1", MI, H / 2, moe * E, "ep_individual"),
    ...fp4Rows("Routed Experts", "experts.w3", MI, H / 2, moe * E, "ep_individual"),
    ...fp4Rows("Routed Experts", "experts.w2", H / 2, MI, moe * E, "ep_individual"),
    ...denseMlpRows("Shared Experts", "shared_experts", H, sharedI, moe, "BF16", 2, "shared"),
    row("KDA", "q_proj.weight", [D, H], "BF16", 2, kda, "tp"),
    row("KDA", "k_proj.weight", [D, H], "BF16", 2, kda, "tp"),
    row("KDA", "v_proj.weight", [D, H], "BF16", 2, kda, "tp"),
    row("KDA", "g_proj.weight", [D, H], "BF16", 2, L, "tp"),
    row("KDA", "b_proj.weight", [kdaHeads, H], "BF16", 2, kda, "tp"),
    row("KDA", "f_a_proj.weight", [kdaDim, H], "BF16", 2, kda),
    row("KDA", "f_b_proj.weight", [D, kdaDim], "BF16", 2, kda, "tp"),
    row("KDA", "q_conv1d.weight", [D, 1, c.linear_attn_config.short_conv_kernel_size], "F32", 4, kda, "tp"),
    row("KDA", "k_conv1d.weight", [D, 1, c.linear_attn_config.short_conv_kernel_size], "F32", 4, kda, "tp"),
    row("KDA", "v_conv1d.weight", [D, 1, c.linear_attn_config.short_conv_kernel_size], "F32", 4, kda, "tp"),
    row("KDA", "A_log", [kdaDim], "F32", 4, kda, "tp"),
    row("KDA", "dt_bias", [D], "F32", 4, kda, "tp"),
    row("KDA", "o_norm.weight", [kdaDim], "F32", 4, kda),
    row("Attention", "o_proj.weight", [H, D], "BF16", 2, L, "otp"),
    row("MLA", "q_a_proj.weight", [qRank, H], "BF16", 2, mla),
    row("MLA", "q_a_layernorm.weight", [qRank], "BF16", 2, mla),
    row("MLA", "q_b_proj.weight", [c.num_attention_heads * (qNope + qRope), qRank], "BF16", 2, mla, "tp"),
    row("MLA", "kv_a_proj_with_mqa.weight", [kvRank + qRope, H], "BF16", 2, mla),
    row("MLA", "kv_a_layernorm.weight", [kvRank], "BF16", 2, mla),
    row("MLA", "kv_b_proj.weight", [c.num_attention_heads * (qNope + vDim), kvRank], "BF16", 2, mla, "tp"),
    row("Vision", "patch_embed.proj.weight", [v.vt_hidden_size, 3, v.patch_size, v.patch_size], "BF16", 2),
    row("Vision", "patch_embed.pos_emb.weight", [v.init_pos_emb_height, v.init_pos_emb_width, v.vt_hidden_size], "BF16", 2),
    row("Vision", "encoder.wqkv.weight", [3 * v.qkv_hidden_size, v.vt_hidden_size], "BF16", 2, v.vt_num_hidden_layers),
    row("Vision", "encoder.wo.weight", [v.vt_hidden_size, v.qkv_hidden_size], "BF16", 2, v.vt_num_hidden_layers),
    row("Vision", "encoder.mlp.fc0.weight", [v.vt_intermediate_size, v.vt_hidden_size], "BF16", 2, v.vt_num_hidden_layers),
    row("Vision", "encoder.mlp.fc1.weight", [v.vt_hidden_size, v.vt_intermediate_size], "BF16", 2, v.vt_num_hidden_layers),
    row("Vision", "encoder.norm0.weight", [v.vt_hidden_size], "BF16", 2, v.vt_num_hidden_layers),
    row("Vision", "encoder.norm1.weight", [v.vt_hidden_size], "BF16", 2, v.vt_num_hidden_layers),
    row("Vision", "final_layernorm.weight", [v.vt_hidden_size], "BF16", 2),
    row("MM Projector", "proj.0.weight", [v.vt_intermediate_size, v.vt_intermediate_size], "BF16", 2),
    row("MM Projector", "proj.2.weight", [H, v.vt_intermediate_size], "BF16", 2),
    row("MM Projector", "post_norm.weight", [H], "BF16", 2),
  ];
  return { rows, facts: { H, vocab: c.vocab_size, layers: L, attentionLayers: L, kdaLayers: kda, mlaLayers: mla, tpAttentionLayers: L, dcpAttentionLayers: mla, denseLayers: dense, moeLayers: moe, heads: c.num_attention_heads, headDim: vDim, kvHeads: c.num_key_value_heads, kvRank, ropeDim: qRope, experts: E, topK: c.num_experts_per_token, indexTopK: 0, indexerLayers: 0 } };
}

function dsparkSpec(c) {
  const H = c.hidden_size;
  const L = c.num_hidden_layers;
  const qWidth = c.num_attention_heads * c.head_dim;
  const kvWidth = c.num_key_value_heads * c.head_dim;
  const markovRank = c.markov_rank;
  const rows = [
    row("Input", "fc.weight", [H, H * L], "BF16", 2, 1, "tp"),
    row("Input", "hidden_norm.weight", [H], "BF16", 2),
    row("Attention", "q_proj.weight", [qWidth, H], "BF16", 2, L, "tp"),
    row("Attention", "k_proj.weight", [kvWidth, H], "BF16", 2, L, "tp"),
    row("Attention", "v_proj.weight", [kvWidth, H], "BF16", 2, L, "tp"),
    row("Attention", "o_proj.weight", [H, qWidth], "BF16", 2, L, "otp"),
    row("Attention", "q_norm.weight", [c.head_dim], "BF16", 2, L),
    row("Attention", "k_norm.weight", [c.head_dim], "BF16", 2, L),
    row("Backbone", "input_layernorm.weight", [H], "BF16", 2, L),
    row("Backbone", "post_attention_layernorm.weight", [H], "BF16", 2, L),
    ...denseMlpRows("Dense MLP", "mlp", H, c.intermediate_size, L),
    row("Output", "norm.weight", [H], "BF16", 2),
    row("Markov Head", "markov_w1.weight", [c.vocab_size, markovRank], "BF16", 2, 1, "embedding"),
    row("Markov Head", "markov_w2.weight", [c.vocab_size, markovRank], "BF16", 2, 1, "lmhead"),
    row("Confidence Head", "confidence_head.proj.weight", [1, H + markovRank], "BF16", 2),
    row("Confidence Head", "confidence_head.proj.bias", [1], "BF16", 2),
  ];
  return { rows, facts: { H, vocab: c.vocab_size, layers: L, attentionLayers: L, tpAttentionLayers: L, dcpAttentionLayers: L, denseLayers: L, moeLayers: 0, heads: c.num_attention_heads, headDim: c.head_dim, kvHeads: c.num_key_value_heads, kvRank: 0, ropeDim: 0, experts: 0, topK: 0, indexTopK: 0, indexerLayers: 0 } };
}

function deepseekSpec(c) {
  const H = c.hidden_size;
  const L = c.num_hidden_layers;
  const heads = c.num_attention_heads;
  const headDim = c.head_dim;
  const qRank = c.q_lora_rank;
  const oRank = c.o_lora_rank;
  const groups = c.o_groups;
  const E = c.n_routed_experts;
  const I = c.moe_intermediate_size;
  const mtp = c.dspark_target_layer_ids.length;
  const ratios = c.compress_ratios.slice(0, L);
  const c4 = ratios.filter((x) => x === 4).length;
  const c128 = ratios.filter((x) => x === 128).length;
  const hcMix = (2 + c.hc_mult) * c.hc_mult;
  const hcDim = c.hc_mult * H;
  const rows = [
    row("Embedding", "embed.weight", [c.vocab_size, H], "BF16", 2, 1, "embedding"),
    row("LM Head", "head.weight", [c.vocab_size, H], "BF16", 2, 1, "lmhead"),
    row("Backbone", "norm.weight", [H], "BF16", 2),
    row("Backbone", "attn_norm.weight", [H], "BF16", 2, L),
    row("Backbone", "ffn_norm.weight", [H], "BF16", 2, L),
    row("DSA", "attn_sink", [heads], "F32", 4, L, "dsa_tp"),
    ...fp8Rows("DSA", "wq_a", qRank, H, L, "replicated"),
    row("DSA", "q_norm.weight", [qRank], "BF16", 2, L),
    ...fp8Rows("DSA", "wq_b", heads * headDim, qRank, L, "dsa_tp"),
    ...fp8Rows("DSA", "wkv", headDim, H, L, "replicated"),
    row("DSA", "kv_norm.weight", [headDim], "BF16", 2, L),
    ...fp8Rows("DSA", "wo_a", groups * oRank, heads * headDim / groups, L, "otp"),
    ...fp8Rows("DSA", "wo_b", H, groups * oRank, L, "otp"),
    row("Compressor C4", "compressor.ape", [4, 2 * headDim], "F32", 4, c4),
    row("Compressor C4", "compressor.norm.weight", [headDim], "BF16", 2, c4),
    row("Compressor C4", "compressor.wkv.weight", [2 * headDim, H], "BF16", 2, c4),
    row("Compressor C4", "compressor.wgate.weight", [2 * headDim, H], "BF16", 2, c4),
    row("Compressor C128", "compressor.ape", [128, headDim], "F32", 4, c128),
    row("Compressor C128", "compressor.norm.weight", [headDim], "BF16", 2, c128),
    row("Compressor C128", "compressor.wkv.weight", [headDim, H], "BF16", 2, c128),
    row("Compressor C128", "compressor.wgate.weight", [headDim, H], "BF16", 2, c128),
    ...fp8Rows("Indexer", "indexer.wq_b", c.index_n_heads * c.index_head_dim, qRank, c4, "replicated"),
    row("Indexer", "indexer.weights_proj.weight", [c.index_n_heads, H], "BF16", 2, c4),
    row("Indexer", "indexer.compressor.ape", [4, 2 * c.index_head_dim], "F32", 4, c4),
    row("Indexer", "indexer.compressor.norm.weight", [c.index_head_dim], "BF16", 2, c4),
    row("Indexer", "indexer.compressor.wkv.weight", [2 * c.index_head_dim, H], "BF16", 2, c4),
    row("Indexer", "indexer.compressor.wgate.weight", [2 * c.index_head_dim, H], "BF16", 2, c4),
    row("MoE Router", "ffn.gate.weight", [E, H], "BF16", 2, L),
    row("MoE Router", "ffn.gate.bias", [E], "F32", 4, L - mtp),
    row("DSpark Routing", "ffn.gate.tid2eid", [c.vocab_size, c.num_experts_per_tok], "I64", 8, mtp),
    ...fp4Rows("Routed Experts", "ffn.experts.w1", I, H, L * E, "ep_individual"),
    ...fp4Rows("Routed Experts", "ffn.experts.w3", I, H, L * E, "ep_individual"),
    ...fp4Rows("Routed Experts", "ffn.experts.w2", H, I, L * E, "ep_individual"),
    ...fp8Rows("Shared Experts", "ffn.shared_experts.w1", I, H, L, "shared"),
    ...fp8Rows("Shared Experts", "ffn.shared_experts.w3", I, H, L, "shared"),
    ...fp8Rows("Shared Experts", "ffn.shared_experts.w2", H, I, L, "shared"),
    row("HC", "hc_attn_fn", [hcMix, hcDim], "F32", 4, L),
    row("HC", "hc_ffn_fn", [hcMix, hcDim], "F32", 4, L),
    row("HC", "hc_attn_base", [hcMix], "F32", 4, L),
    row("HC", "hc_ffn_base", [hcMix], "F32", 4, L),
    row("HC", "hc_attn_scale", [3], "F32", 4, L),
    row("HC", "hc_ffn_scale", [3], "F32", 4, L),
  ];
  const mtpBase = deepseekMtpRows(c, mtp, hcMix, hcDim);
  rows.push(...mtpBase);
  return { rows, facts: { H, vocab: c.vocab_size, layers: L, attentionLayers: L, tpAttentionLayers: L, dcpAttentionLayers: 0, denseLayers: 0, moeLayers: L, heads, headDim, kvHeads: 1, kvRank: headDim, ropeDim: c.qk_rope_head_dim, experts: E, topK: c.num_experts_per_tok, indexTopK: c.index_topk, indexerLayers: c4 } };
}

function deepseekMtpRows(c, mtp, hcMix, hcDim) {
  const H = c.hidden_size;
  const heads = c.num_attention_heads;
  const D = c.head_dim;
  const qRank = c.q_lora_rank;
  const oRank = c.o_lora_rank;
  const groups = c.o_groups;
  const E = c.n_routed_experts;
  const I = c.moe_intermediate_size;
  return [
    row("MTP", "mtp.attn_norm.weight", [H], "BF16", 2, mtp),
    row("MTP", "mtp.ffn_norm.weight", [H], "BF16", 2, mtp),
    row("MTP", "mtp.attn.attn_sink", [heads], "F32", 4, mtp, "dsa_tp"),
    ...fp8Rows("MTP", "mtp.attn.wq_a", qRank, H, mtp, "replicated"),
    row("MTP", "mtp.attn.q_norm.weight", [qRank], "BF16", 2, mtp),
    ...fp8Rows("MTP", "mtp.attn.wq_b", heads * D, qRank, mtp, "dsa_tp"),
    ...fp8Rows("MTP", "mtp.attn.wkv", D, H, mtp, "replicated"),
    row("MTP", "mtp.attn.kv_norm.weight", [D], "BF16", 2, mtp),
    ...fp8Rows("MTP", "mtp.attn.wo_a", groups * oRank, heads * D / groups, mtp, "otp"),
    ...fp8Rows("MTP", "mtp.attn.wo_b", H, groups * oRank, mtp, "otp"),
    row("MTP", "mtp.ffn.gate.weight", [E, H], "BF16", 2, mtp),
    row("MTP", "mtp.ffn.gate.bias", [E], "F32", 4, mtp),
    ...fp4Rows("MTP Routed Experts", "mtp.ffn.experts.w1", I, H, mtp * E, "ep_individual"),
    ...fp4Rows("MTP Routed Experts", "mtp.ffn.experts.w3", I, H, mtp * E, "ep_individual"),
    ...fp4Rows("MTP Routed Experts", "mtp.ffn.experts.w2", H, I, mtp * E, "ep_individual"),
    ...fp8Rows("MTP Shared Experts", "mtp.ffn.shared_experts.w1", I, H, mtp, "shared"),
    ...fp8Rows("MTP Shared Experts", "mtp.ffn.shared_experts.w3", I, H, mtp, "shared"),
    ...fp8Rows("MTP Shared Experts", "mtp.ffn.shared_experts.w2", H, I, mtp, "shared"),
    row("MTP HC", "mtp.hc_attn_fn", [hcMix, hcDim], "F32", 4, mtp),
    row("MTP HC", "mtp.hc_ffn_fn", [hcMix, hcDim], "F32", 4, mtp),
    row("MTP HC", "mtp.hc_attn_base", [hcMix], "F32", 4, mtp),
    row("MTP HC", "mtp.hc_ffn_base", [hcMix], "F32", 4, mtp),
    row("MTP HC", "mtp.hc_attn_scale", [3], "F32", 4, mtp),
    row("MTP HC", "mtp.hc_ffn_scale", [3], "F32", 4, mtp),
    row("MTP", "mtp.main_norm.weight", [H], "BF16", 2),
    ...fp8Rows("MTP", "mtp.main_proj", H, 3 * H, 1, "replicated"),
    row("MTP", "mtp.norm.weight", [H], "BF16", 2),
    row("MTP Markov Head", "mtp.markov_w1.weight", [c.vocab_size, c.dspark_markov_rank], "BF16", 2, 1, "embedding"),
    row("MTP Markov Head", "mtp.markov_w2.weight", [c.vocab_size, c.dspark_markov_rank], "BF16", 2, 1, "lmhead"),
    row("MTP Confidence Head", "mtp.confidence_head.proj.weight", [1, H + c.dspark_markov_rank], "BF16", 2),
    row("MTP HC Head", "mtp.hc_head_fn", [4, 4 * H], "F32", 4),
    row("MTP HC Head", "mtp.hc_head_base", [4], "F32", 4),
    row("MTP HC Head", "mtp.hc_head_scale", [1], "F32", 4),
    row("HC Head", "hc_head_fn", [4, 4 * H], "F32", 4),
    row("HC Head", "hc_head_base", [4], "F32", 4),
    row("HC Head", "hc_head_scale", [1], "F32", 4),
  ];
}

function glmSpec(c) {
  const H = c.hidden_size;
  const baseLayers = c.num_hidden_layers;
  const mtp = c.num_nextn_predict_layers;
  const L = baseLayers + mtp;
  const dense = c.mlp_layer_types.filter((x) => x === "dense").length;
  const moe = L - dense;
  const E = c.n_routed_experts;
  const I = c.moe_intermediate_size;
  let indexers = 0;
  for (let id = 0; id < L; id++) {
    const isMtp = id >= baseLayers;
    const skip = Math.max(id - c.index_skip_topk_offset + 1, 0) % c.index_topk_freq !== 0;
    if (!skip || isMtp) indexers++;
  }
  const f32Scale = ["F32", 4];
  const rows = [
    row("Embedding", "model.embed_tokens.weight", [c.vocab_size, H], "BF16", 2, 1, "embedding"),
    row("LM Head", "lm_head.weight", [c.vocab_size, H], "BF16", 2, 1, "lmhead"),
    row("Backbone", "model.norm.weight", [H], "BF16", 2),
    row("Backbone", "input_layernorm.weight", [H], "BF16", 2, L),
    row("Backbone", "post_attention_layernorm.weight", [H], "BF16", 2, L),
    ...fp8Rows("Dense MLP", "mlp.gate_proj", c.intermediate_size, H, dense, "tp", ...f32Scale),
    ...fp8Rows("Dense MLP", "mlp.up_proj", c.intermediate_size, H, dense, "tp", ...f32Scale),
    ...fp8Rows("Dense MLP", "mlp.down_proj", H, c.intermediate_size, dense, "tp", ...f32Scale),
    row("MoE Router", "mlp.gate.weight", [E, H], "BF16", 2, moe),
    row("MoE Router", "mlp.gate.e_score_correction_bias", [E], "F32", 4, moe),
    ...fp8Rows("Routed Experts", "mlp.experts.gate_proj", I, H, moe * E, "ep_individual", ...f32Scale),
    ...fp8Rows("Routed Experts", "mlp.experts.up_proj", I, H, moe * E, "ep_individual", ...f32Scale),
    ...fp8Rows("Routed Experts", "mlp.experts.down_proj", H, I, moe * E, "ep_individual", ...f32Scale),
    ...fp8Rows("Shared Experts", "mlp.shared_experts.gate_proj", I, H, moe, "shared", ...f32Scale),
    ...fp8Rows("Shared Experts", "mlp.shared_experts.up_proj", I, H, moe, "shared", ...f32Scale),
    ...fp8Rows("Shared Experts", "mlp.shared_experts.down_proj", H, I, moe, "shared", ...f32Scale),
    ...fp8Rows("MLA", "q_a_proj", c.q_lora_rank, H, L, "replicated", ...f32Scale),
    row("MLA", "q_a_layernorm.weight", [c.q_lora_rank], "BF16", 2, L),
    ...fp8Rows("MLA", "q_b_proj", c.num_attention_heads * c.qk_head_dim, c.q_lora_rank, L, "dsa_tp", ...f32Scale),
    ...fp8Rows("MLA", "kv_a_proj_with_mqa", c.kv_lora_rank + c.qk_rope_head_dim, H, L, "replicated", ...f32Scale),
    row("MLA", "kv_a_layernorm.weight", [c.kv_lora_rank], "BF16", 2, L),
    ...fp8Rows("MLA", "kv_b_proj", c.num_attention_heads * (c.qk_nope_head_dim + c.v_head_dim), c.kv_lora_rank, L, "dsa_tp", ...f32Scale),
    ...fp8Rows("MLA", "o_proj", H, c.num_attention_heads * c.v_head_dim, L, "otp", ...f32Scale),
    row("Indexer", "indexer.k_norm.weight", [c.index_head_dim], "BF16", 2, indexers),
    row("Indexer", "indexer.k_norm.bias", [c.index_head_dim], "BF16", 2, indexers),
    ...fp8Rows("Indexer", "indexer.wk", c.index_head_dim, H, indexers, "replicated", ...f32Scale),
    ...fp8Rows("Indexer", "indexer.wq_b", c.index_n_heads * c.index_head_dim, c.q_lora_rank, indexers, "replicated", ...f32Scale),
    row("Indexer", "indexer.weights_proj.weight", [c.index_n_heads, H], "BF16", 2, indexers),
    row("MTP", "eh_proj.weight", [H, 2 * H], "BF16", 2),
    row("MTP", "enorm.weight", [H], "BF16", 2),
    row("MTP", "hnorm.weight", [H], "BF16", 2),
    row("MTP", "shared_head.norm.weight", [H], "BF16", 2),
  ];
  return { rows, facts: { H, vocab: c.vocab_size, layers: L, attentionLayers: L, tpAttentionLayers: L, dcpAttentionLayers: 0, denseLayers: dense, moeLayers: moe, heads: c.num_attention_heads, headDim: c.v_head_dim, kvHeads: c.num_key_value_heads, kvRank: c.kv_lora_rank, ropeDim: c.qk_rope_head_dim, experts: E, topK: c.num_experts_per_tok, indexTopK: c.index_topk, indexerLayers: indexers } };
}

function qwenSpec(c) {
  const H = c.hidden_size;
  const L = c.num_hidden_layers;
  const full = c.layer_types.filter((x) => x === "full_attention").length;
  const linear = L - full;
  const E = c.num_experts;
  const I = c.moe_intermediate_size;
  const qWidth = c.num_attention_heads * c.head_dim;
  const kvWidth = c.num_key_value_heads * c.head_dim;
  const keyWidth = c.linear_num_key_heads * c.linear_key_head_dim;
  const valueWidth = c.linear_num_value_heads * c.linear_value_head_dim;
  const convWidth = 2 * keyWidth + valueWidth;
  const rows = [
    row("Embedding", "model.embed_tokens.weight", [c.vocab_size, H], "BF16", 2, 1, "embedding"),
    row("LM Head", "lm_head.weight", [c.vocab_size, H], "BF16", 2, 1, "lmhead"),
    row("Backbone", "model.norm.weight", [H], "BF16", 2),
    row("Backbone", "input_layernorm.weight", [H], "BF16", 2, L),
    row("Backbone", "post_attention_layernorm.weight", [H], "BF16", 2, L),
    row("Full Attention", "q_proj.weight", [2 * qWidth, H], "BF16", 2, full, "tp"),
    row("Full Attention", "k_proj.weight", [kvWidth, H], "BF16", 2, full, "tp"),
    row("Full Attention", "v_proj.weight", [kvWidth, H], "BF16", 2, full, "tp"),
    row("Full Attention", "o_proj.weight", [H, qWidth], "BF16", 2, full, "otp"),
    row("Full Attention", "q_norm.weight", [c.head_dim], "BF16", 2, full),
    row("Full Attention", "k_norm.weight", [c.head_dim], "BF16", 2, full),
    row("Gated DeltaNet", "conv1d.weight", [convWidth, 1, c.linear_conv_kernel_dim], "BF16", 2, linear, "tp"),
    row("Gated DeltaNet", "in_proj_qkv.weight", [convWidth, H], "BF16", 2, linear, "tp"),
    row("Gated DeltaNet", "in_proj_z.weight", [valueWidth, H], "BF16", 2, linear, "tp"),
    row("Gated DeltaNet", "in_proj_a.weight", [c.linear_num_value_heads, H], "BF16", 2, linear, "tp"),
    row("Gated DeltaNet", "in_proj_b.weight", [c.linear_num_value_heads, H], "BF16", 2, linear, "tp"),
    row("Gated DeltaNet", "out_proj.weight", [H, valueWidth], "BF16", 2, linear, "otp"),
    row("Gated DeltaNet", "A_log", [c.linear_num_value_heads], "BF16", 2, linear, "tp"),
    row("Gated DeltaNet", "dt_bias", [c.linear_num_value_heads], "BF16", 2, linear, "tp"),
    row("Gated DeltaNet", "norm.weight", [c.linear_value_head_dim], "BF16", 2, linear),
    row("MoE Router", "mlp.gate.weight", [E, H], "BF16", 2, L),
    row("MoE Router", "mlp.shared_expert_gate.weight", [1, H], "BF16", 2, L),
    row("Routed Experts", "mlp.experts.gate_up_proj", [E, 2 * I, H], "BF16", 2, L, "ep_tensor"),
    row("Routed Experts", "mlp.experts.down_proj", [E, H, I], "BF16", 2, L, "ep_tensor"),
    ...denseMlpRows("Shared Expert", "mlp.shared_expert", H, c.shared_expert_intermediate_size, L, "BF16", 2, "shared"),
    row("MTP", "mtp.fc.weight", [H, 2 * H], "BF16", 2, 1, "tp"),
    row("MTP", "mtp.input_layernorm.weight", [H], "BF16", 2),
    row("MTP", "mtp.post_attention_layernorm.weight", [H], "BF16", 2),
    row("MTP", "mtp.norm.weight", [H], "BF16", 2),
    row("MTP", "mtp.pre_fc_norm_embedding.weight", [H], "BF16", 2),
    row("MTP", "mtp.pre_fc_norm_hidden.weight", [H], "BF16", 2),
    row("MTP Full Attention", "mtp.q_proj.weight", [2 * qWidth, H], "BF16", 2, 1, "tp"),
    row("MTP Full Attention", "mtp.k_proj.weight", [kvWidth, H], "BF16", 2, 1, "tp"),
    row("MTP Full Attention", "mtp.v_proj.weight", [kvWidth, H], "BF16", 2, 1, "tp"),
    row("MTP Full Attention", "mtp.o_proj.weight", [H, qWidth], "BF16", 2, 1, "otp"),
    row("MTP Full Attention", "mtp.q_norm.weight", [c.head_dim], "BF16", 2),
    row("MTP Full Attention", "mtp.k_norm.weight", [c.head_dim], "BF16", 2),
    row("MTP Router", "mtp.mlp.gate.weight", [E, H], "BF16", 2),
    row("MTP Router", "mtp.mlp.shared_expert_gate.weight", [1, H], "BF16", 2),
    row("MTP Routed Experts", "mtp.mlp.experts.gate_up_proj", [E, 2 * I, H], "BF16", 2, 1, "ep_tensor"),
    row("MTP Routed Experts", "mtp.mlp.experts.down_proj", [E, H, I], "BF16", 2, 1, "ep_tensor"),
    ...denseMlpRows("MTP Shared Expert", "mtp.mlp.shared_expert", H, c.shared_expert_intermediate_size, 1, "BF16", 2, "shared"),
  ];
  return { rows, facts: { H, vocab: c.vocab_size, layers: L, attentionLayers: L, fullAttentionLayers: full, linearAttentionLayers: linear, tpAttentionLayers: L, dcpAttentionLayers: full, denseLayers: 0, moeLayers: L, heads: c.num_attention_heads, headDim: c.head_dim, kvHeads: c.num_key_value_heads, kvRank: 0, ropeDim: 0, experts: E, topK: c.num_experts_per_tok, indexTopK: 0, indexerLayers: 0 } };
}

export function modelSpec(model, config) {
  if (model.profile === "deepseek_v41") return v41WeightSpec(config);
  if (model.profile === "kimi") return kimiSpec(config);
  if (model.profile === "kimi_dspark") return dsparkSpec(config);
  if (model.profile === "deepseek_v4") return deepseekSpec(config);
  if (model.profile === "glm53") return glmSpec(config);
  if (model.profile === "qwen38") return qwenSpec(config);
  throw new Error(`Unsupported profile: ${model.profile}`);
}

export function shapeText(dims) {
  return `[${dims.join(", ")}]`;
}

export function elementCount(dims) {
  return dims.reduce((product, dim) => product * dim, 1);
}
