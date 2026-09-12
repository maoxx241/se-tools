# Calculation methodology

中文场景说明见 [SESSION.md](SESSION.md)，HCCL API 基数、算法假设与 Profiling 对照见 [HCCL.md](HCCL.md)。以下公式是模型输入下的计算，不代表实际链路测量。

本页保留历史模型的方法。2026-09-12 新增的 **V4.1 Flash** 使用独立的[架构/运行时公式](DEEPSEEK-V4.1.md)，包括源层 KV 共享、SP 共享专家复制和 Engram。旧 V4 的 C4/C128 或 Indexer 通信行不适用于 V4.1。

## Weight model

The generator uses each model's adjacent `config.json` and the pinned model implementation to enumerate tensors. Historical model rows do not validate Safetensors headers. V4.1's config-derived rows are additionally checked by the portable tests against a complete, separately collected header snapshot; generation itself remains offline.

Each workbook row keeps projection weights, normalization weights, quantized weights, and quantization scales separate. In particular, `q_a_proj`, `q_a_layernorm`, and block scales never share one dtype or byte-width input.

```text
single tensor elements = product(shape dimensions)
total weight GiB       = elements * tensor count * bytes per element / 2^30
single-rank weight GiB = total weight GiB / partition count
```

`字节/元素` is an editable Excel cell on every row. FP4 data uses `0.5`; FP8 data and E8M0 scales use `1`; BF16 uses `2`; F32 uses `4` by default. Changing one row never changes unrelated norms or scales.

DP replicates model weights. TP, EP, shared-expert TP/DP, O Proj TP, Embedding TP, and LM Head TP select the partition count. For DeepSeek-V4 and GLM-5.3, DSA CP replicates the DSA Q-head projection that ordinary TP would shard; Kimi and Qwen retain ordinary TP.

## Step tokens

```text
Prefill step tokens = min(max_num_batched_tokens,
                          max_num_seqs * average_input_tokens)

Decode step tokens = min(max_num_batched_tokens,
                         max_num_seqs * (speculative_steps + 1))
```

Prefill and Decode have independent input panels and formulas.

## Communication rows

Every communication row names its parallel strategy, module, call position, collective, local input elements, bytes per element, group size, and calls per step. A row with `次数/Step = 0` is disabled by the current strategy.

For group size `p` and local input payload `B` bytes:

```text
Ring AllReduce sent bytes       = B * 2 * (p - 1) / p
Ring AllGather sent bytes       = B * (p - 1)
Ring ReduceScatter sent bytes   = B * (p - 1) / p
Equal-split AllToAll sent bytes = B * (p - 1) / p
```

For EP AllToAllV, the input is a local cross-rank token-expert fraction. Its default `(EP-1)/EP` is a uniform-routing expectation. Multiplying one rank by EP also assumes balanced input and routing; use the full send-count matrix in `lib/collectives.mjs` for nonuniform traffic. Do not apply the cross-rank fraction twice. Dispatch/combine rows in the historical model include hidden tensors only, with equal widths; routing metadata and quantization scales need separate runtime evidence.

PCP and DCP default to group size `1` (disabled). SP defaults to enabled. DeepSeek-V4 and GLM-5.3 enable their model-specific DSA CP mapping by default; Kimi and Qwen do not expose that switch.

The workbook covers the code-backed communication calls currently represented by the generator: embedding and LM-head TP, attention SP/O TP, routed-expert EP, MoE DP, shared-expert TP, PCP cache/MoE collectives, DCP query/LSE/output collectives, and DSA CP layout restoration. It does not include workspace, allocator residency, KV-cache capacity, offload hit/miss traffic, or physical link-algorithm selection.

## Aggregation and scope

Each row's group total is for that named group, not the full deployment. Different TP, EP and DP groups must not be blindly added or multiplied by DP again. Newly generated workbooks leave the group-total sum blank. The per-rank event sum assumes the same representative rank participates in the modeled calls.

The fixed communication template also repeats Attention as a module view of SP/DSA events. These rows overlap and are not an additive event ledger. P→D cache is a separate per-transfer calculation for one source DP batch of 16 requests, not a per-step collective. Its formulas include destination replicas and allocated recurrent/conv state under the historical assumptions. A connector may transfer a different subset.

The generators expose theoretical arithmetic choices, not a complete validator for all vLLM/HCCL parallelism combinations. Confirm model-specific sharding, PCP/DCP/SP compatibility, communication dtype support and selected runtime paths before treating a changed configuration as deployable.
