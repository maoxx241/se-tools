function tensor(group, name, shape, layers, rule = 'Replicated', dtype = 'BF16', extra = {}) {
  return { group, name, shape, layerIds: layers, rule, dtype, shared: rule === 'Shared TP', ...extra };
}

export function kimiMemorySpec(config) {
  if (config.model_type !== 'kimi_k3') throw new Error(`Unsupported public Kimi model_type: ${config.model_type}`);
  const c = config.text_config, v = config.vision_config, linear = c.linear_attn_config;
  const ids = Array.from({ length: c.num_hidden_layers }, (_, i) => i + 1);
  const kda = linear.kda_layers, mla = linear.full_attn_layers;
  if (new Set([...kda, ...mla]).size !== ids.length || kda.length + mla.length !== ids.length
      || ids.some(id => ![...kda, ...mla].includes(id))) throw new Error('Attention layer lists must partition all decoder layers (1-based)');
  const dense = ids.slice(0, c.first_k_dense_replace), moe = ids.slice(c.first_k_dense_replace);
  const H = c.hidden_size, D = linear.num_heads * linear.head_dim, I = c.intermediate_size;
  const R = c.routed_expert_hidden_size, M = c.moe_intermediate_size, S = M * c.num_shared_experts;
  const Q = c.q_lora_rank, K = c.kv_lora_rank, attn = 'Gated MLA';
  const rows = [
    tensor('Embedding / Output', 'embed_tokens.weight', [c.vocab_size, H], [], 'Embedding TP', 'BF16', { placement: 'first' }),
    tensor('Embedding / Output', 'lm_head.weight', [c.vocab_size, H], [], 'LM head TP', 'BF16', { placement: 'last' }),
    ...['input_layernorm.weight', 'post_attention_layernorm.weight', 'self_attention_res_norm.weight', 'mlp_res_norm.weight']
      .map(name => tensor('Decoder common', name, [H], ids)),
    ...['self_attention_res_proj.weight', 'mlp_res_proj.weight'].map(name => tensor('Decoder common', name, [1, H], ids)),
    ...['final_norm.weight', 'output_attn_res_norm.weight'].map(name => tensor('Decoder common', name, [H], [], 'Replicated', 'BF16', { placement: 'last' })),
    tensor('Decoder common', 'output_attn_res_proj.weight', [1, H], [], 'Replicated', 'BF16', { placement: 'last' }),
    ...['gate_proj.weight', 'up_proj.weight', 'down_proj.weight'].map(name =>
      tensor('Dense MLP', name, name.startsWith('down') ? [H, I] : [I, H], dense, 'Other TP')),
    ...['q_proj', 'k_proj', 'v_proj'].map(name => tensor('KDA', name, [D, H], kda, 'KDA TP')),
    ...['q_conv1d.weight', 'k_conv1d.weight', 'v_conv1d.weight'].map(name =>
      tensor('KDA', name, [D, 1, linear.short_conv_kernel_size], kda, 'KDA TP', 'FP32')),
    tensor('KDA', 'A_log', [linear.num_heads], kda, 'KDA TP', 'FP32'),
    tensor('KDA', 'f_a_proj', [linear.head_dim, H], kda),
    tensor('KDA', 'f_b_proj', [D, linear.head_dim], kda, 'KDA TP'),
    tensor('KDA', 'dt_bias', [D], kda, 'KDA TP', 'FP32'),
    tensor('KDA', 'b_proj', [linear.num_heads, H], kda, 'KDA TP'),
    tensor('KDA', 'g_proj', [D, H], kda, 'KDA TP'),
    tensor('KDA', 'o_norm.weight', [linear.head_dim], kda),
    tensor('KDA', 'o_proj', [H, D], kda, 'KDA O TP'),
    tensor(attn, 'q_a_proj', [Q, H], mla),
    tensor(attn, 'q_a_layernorm.weight', [Q], mla),
    tensor(attn, 'q_b_proj', [c.num_attention_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim), Q], mla, 'DSA TP'),
    tensor(attn, 'kv_a_proj_with_mqa', [K + c.qk_rope_head_dim, H], mla),
    tensor(attn, 'kv_a_layernorm.weight', [K], mla),
    tensor(attn, 'kv_b_proj', [c.num_attention_heads * (c.qk_nope_head_dim + c.v_head_dim), K], mla, 'DSA TP'),
    tensor(attn, 'g_proj', [D, H], mla, 'DSA TP'),
    tensor(attn, 'o_proj', [H, D], mla, 'DSA O TP'),
  ];
  rows.push(
    tensor('MoE', 'router.weight', [c.num_experts, H], moe),
    tensor('MoE', 'e_score_correction_bias', [c.num_experts], moe),
    tensor('MoE', 'routed_expert_down_proj', [R, H], moe),
    tensor('MoE', 'routed_expert_norm.weight', [R], moe),
    tensor('MoE', 'routed_expert_up_proj', [H, R], moe),
    ...['w1', 'w3', 'w2'].map(name => tensor('MoE', `experts.*.${name}`, name === 'w2' ? [R, M] : [M, R], moe, 'EP', 'MXFP4')),
    ...['gate_proj', 'up_proj', 'down_proj'].map(name => tensor('MoE', `shared_experts.${name}`,
      name === 'down_proj' ? [H, S] : [S, H], moe, 'Shared TP')),
    tensor('ViT patch', 'patch_embed.proj.weight', [v.vt_hidden_size, 3, v.patch_size, v.patch_size], [], 'Replicated', 'BF16', { placement: 'first' }),
    tensor('ViT patch', 'patch_embed.pos_emb.weight', [v.init_pos_emb_height, v.init_pos_emb_width, v.vt_hidden_size], [], 'Replicated', 'BF16', { placement: 'first' }),
    tensor('ViT blocks', 'blocks.*.{norm0,norm1}.weight', [v.vt_hidden_size], [], 'Replicated', 'BF16', { placement: 'first', instances: 2 * v.vt_num_hidden_layers }),
    ...[
      ['blocks.*.wqkv.weight', [3 * v.qkv_hidden_size, v.vt_hidden_size]],
      ['blocks.*.wo.weight', [v.vt_hidden_size, v.qkv_hidden_size]],
      ['blocks.*.mlp.fc0.weight', [v.vt_intermediate_size, v.vt_hidden_size]],
      ['blocks.*.mlp.fc1.weight', [v.vt_hidden_size, v.vt_intermediate_size]],
    ].map(([name, shape]) => tensor('ViT blocks', name, shape, [], 'Replicated', 'BF16', { placement: 'first', instances: v.vt_num_hidden_layers })),
    tensor('ViT final', 'encoder.final_layernorm.weight', [v.vt_hidden_size], [], 'Replicated', 'BF16', { placement: 'first' }),
    ...[
      ['proj.0.weight', [v.vt_intermediate_size, v.vt_intermediate_size]],
      ['proj.2.weight', [H, v.vt_intermediate_size]],
      ['post_norm.weight', [H]],
    ].map(([name, shape]) => tensor('MM projector', name, shape, [], 'Replicated', 'BF16', { placement: 'first' })),
  );
  const facts = {
    layers: ids.length, kdaLayers: kda, attentionLayers: mla, denseLayers: dense, moeLayers: moe,
    experts: c.num_experts, sharedExperts: c.num_shared_experts, kdaHeads: linear.num_heads,
    headDim: linear.head_dim, attentionHeads: c.num_attention_heads, kvRank: K, ropeDim: c.qk_rope_head_dim,
    indexDim: 0, indexHeads: 0,
    convKernel: linear.short_conv_kernel_size, visionLayers: v.vt_num_hidden_layers,
    hasDSA: false, maxTokens: c.max_position_embeddings,
  };
  return { rows, facts, recipes: ['checkpoint'], evidence: 'Public archived model config/code; not checkpoint-header verification' };
}
