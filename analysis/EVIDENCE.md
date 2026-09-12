# Evidence registry

This file records the session's historical model/runtime revisions. They are separate from the HCCL revision inspected on 2026-09-12; no matching CANN installation or NPU execution was verified during extraction.

The Excel files intentionally do not contain a source column. This registry records the pinned code used by the generator.

## Model inputs

Each model directory contains its archived `config.json`. The six workbook models have the historical Hugging Face revisions listed below; the two additional case inputs have separate provenance. `scripts/model-analysis-specs.mjs` turns dimensions into explicit tensor shapes and repetition counts. No checkpoint tensor index or Safetensors header is required.

Six workbook model revisions are listed in [model-catalog.mjs](../lib/model-catalog.mjs). The original JSON bytes are archived under `models/`; their current SHA-256 values and the seven original XLSX hashes are in [snapshot-manifest.json](snapshot-manifest.json). Remote model availability was not revalidated during extraction.

Additional communication-case configurations:

- `deepseek-v4-10t/config.json`: user-provided DeepSeek V4 10T attachment from the original analysis. No upstream commit is asserted. It is not the Flash configuration.
- `qwen3.8-27b/config.json`: recovered complete JSON from the original session's read-only command output for [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json). That session fetched `main` without recording a commit. This is a content-hashed historical snapshot, not a revision-pinned upstream file; formatting was normalized during recovery.

Model configs remain attributable to their upstream authors and licenses. The repository does not redistribute model weights or HCCL source code.

## Runtime source revisions

- vLLM: `c8438a3d40168ce1d9eade0dc15ccbe5d27adb68`
- vLLM Ascend: `842b030f8375e630eb639e0560eac7735d04f700`

Relevant source locations:

- `vllm/model_executor/layers/linear.py`: column-parallel and row-parallel weight layouts; row-parallel output reduction.
- `vllm/model_executor/layers/vocab_parallel_embedding.py`: embedding and LM-head TP.
- `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`: Kimi KDA parameter construction.
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`: Qwen Gated DeltaNet dimensions and TP layout.
- `vllm/model_executor/models/qwen3_next.py` and `qwen3_5.py`: Qwen SP attention and MoE placement.
- `vllm/model_executor/models/qwen3_5_mtp.py`: Qwen MTP projection and layer.
- `vllm/model_executor/models/deepseek_v2.py`: GLM MLA, Indexer selection, MTP, and tensor-parallel modules.
- `vllm/v1/attention/ops/pcp.py`: PCP cache-input gathers.
- `vllm/v1/attention/ops/dcp.py`: DCP Query AllGather, LSE AllGather, and output ReduceScatter/AllToAll layouts.
- `vllm_ascend/models/kimi_k3.py`: Kimi-K3 text, MoE, vision, and projector modules.
- `vllm_ascend/models/kimi_k3_dspark.py`: Kimi-K3-DSpark modules.
- `vllm_ascend/models/deepseek_v4.py`: DeepSeek-V4 DSA, compressor, indexer, HC, MoE, and DSA CP weight replication.
- `vllm_ascend/attention/context_parallel/dsa_cp.py`: DSA CP token sharding and TP-head-layout AllToAll.
- `vllm_ascend/attention/context_parallel/sfa_cp.py`: GLM SFA context-parallel data movement.
- `vllm_ascend/ops/linear_op.py`: SP and O Proj TP implementations.
- `vllm_ascend/ops/fused_moe/prepare_finalize.py`: MoE DP/PCP gathers and ReduceScatter.
- `vllm_ascend/ops/fused_moe/token_dispatcher.py`: routed-expert AllToAllV split vectors.

## Communication conventions

The Kimi SP/TP rows follow the previously traced TP8 service path: attention input AllGather, post-O-Proj ReduceScatter, dense-MLP local-token AllReduce, shared-expert gather/scatter, final hidden AllGather, and LM-head collective.

EP AllToAllV is parameterized by the local cross-rank routing fraction because `config.json` cannot determine a runtime split vector. The default formula is the uniform-routing value and remains editable in Excel.

## Exclusions

- Workspace and activation residency.
- KV-cache capacity and recurrent-state capacity.
- KV offload hit/miss traffic.
- Allocator, graph-capture, alignment, and temporary-buffer overhead.
- Runtime-selected physical HCCL algorithm and topology effects.

## HCCL evidence added on 2026-09-12

- Repository: [cann/hccl](https://gitcode.com/cann/hccl).
- Inspected commit: `170ddeec539b4d693028ce6e0cf5c58933e4d46d`.
- [HCCL derivation and corrections](HCCL.md) maps API count semantics, Ring derivation, AlltoAllV self-copy, algorithm selection and Profiling interpretation to pinned source links.
- [hccl-sources.json](hccl-sources.json) records exact paths, inspected line ranges and file hashes. These are read-only source evidence, not a claim that the historical model service used this HCCL commit.

## Verification on extraction

The original six model workbooks passed the existing Artifact Tool input-mutation/formula checks again. The portable verifier checks 15 snapshot hashes, seven valid XLSX archives, 84 saved communication byte values, and blank communication-capability cells. Node tests cover HCCL count conventions, uneven routing, zero/single-rank cases, large integers and the corrected non-SP branch.

These checks establish archive fidelity, portable reproduction and selected formula behavior. They do not establish every model's runtime call placement, checkpoint equivalence, deployed topology compatibility, physical link bytes or throughput.
