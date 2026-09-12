# 五模型的 KV 存取与 EP32 / EP256 规格

这是 [256K 历史案例](../examples/communication-256k/) 的新扩展。交付包含 [可编辑 Excel](../examples/kv-ep-sweep/kv-ep32-ep256.xlsx)、[十档数值](../examples/kv-ep-sweep/results.json)、[公共计算器](../lib/kv-ep-specs.mjs) 和 [源码清单](kv-ep-sources.json)。原始 Excel、config 和历史结果保持原样。

## 口径和默认参数

| 参数 | 本次取值 |
|---|---|
| 模型 | DS V4 10T、V4 Pro、GLM5.3、Kimi K3、Qwen3.8 2.4T |
| 输入 / 请求数 | 262,144 tokens / 每个 P DP 16 请求 |
| P Step | 每 DP 16,384 个有效 token；TP8 + SP，每个 EP Rank 2,048 token |
| D Step | 每 DP 128 个有效 token；DS/GLM 每 Rank 128，Kimi/Qwen 每 Rank 16 |
| P 并行 | TP8；EP32→DP4，EP256→DP32 |
| D 并行 | DS/GLM TP1，DP=EP；Kimi/Qwen TP8，DP=EP/8 |
| KV | A3 布局；主 KV BF16；DS Indexer INT8+FP16 scale；GLM Indexer BF16；递归状态 FP32 |
| Cache block | 128；可改 32/64/128；P/D 相同 |
| Hybrid state | P/D speculative conv 扩展槽数都为 7；一次拉一个有效 recurrent state |
| Connector | MooncakeConnector，无前缀命中，PCP/DCP=1；Kimi/Qwen 按非 aligned 模式 |
| EP P / D | AllToAllV / A3 MC2 DispatchV2 FullMesh + BF16 CombineV2 |
| 通信 dtype | 默认 dispatch BF16 / combine BF16；可切 INT8+每 token FP32 scale / BF16 |
| 负载 | 每个源 Rank token 数相同，目标 Rank 路由负载均衡，排除本 Rank 自拷贝 |

MC2 硬件和量化没有用户指定值时，这里选 A3 BF16 **分析基线**。权重量化不能推出通信量化。主表显式选择 FullMesh 分支，不声称该分支就是部署环境自动选中的分支。

EP 对比的主线是**通信量随 DP 域扩大而变化**，Excel 第一张表优先显示这些数据：

| 阶段 / 模型 | EP32 的 DP | EP256 的 DP | EP32 全组 tokens/Step | EP256 全组 tokens/Step |
|---|---:|---:|---:|---:|
| Prefill / 全部 | 4 | 32 | 65,536 | 524,288 |
| Decode / DS、GLM | 32 | 256 | 4,096 | 32,768 |
| Decode / Kimi、Qwen | 4 | 32 | 512 | 4,096 |

这是“每 DP 固定工作量”的容量比较。除 payload 的工作量增加外，跨 Rank 比例也从 31/32 变成 255/256：纯 payload 全组量因此放大 **255/31≈8.226 倍**，不是仅按比例放大 1.028 倍。Rank 平均量只增加约 2.82%；直方图 AllGather 的控制量还含通信域规模项。反之，若固定**全组** token 总数，增大 DP 后每 DP token 数必须相应减少 8 倍；Excel 可在 P/D tokens/DP 输入中复算该比较。

[All2All prepare/finalize 与 MC2 继承路径](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/ops/fused_moe/prepare_finalize.py#L112) 将每 DP 的 token 按 TP 切片后交给 EP。独立的 DP hidden/router AllGather、DP ReduceScatter 是另一种 AllGather backend 的策略，不应叠加到这次指定的 AllToAllV/MC2 路径。TP 恢复、DP 调度元数据同步不在本次 EP 数据面主表中；本次没有把它们计作零流量结论。

“存”按用户指定的 Prefill **生成数据量**：每个生成的 KV/Indexer 行计一次，加一份阶段末有效辅助状态；包含实际 P TP 副本。SWA 在整个 prompt 中生成的行也计入，所以该数可能远大于阶段末缓存容量。“保留”单列阶段末有效数据，帮助解释这一区别。两者都不计 kernel 中间结果、各 chunk 的状态覆盖写、allocator 填充、缓存池预留或 HBM 事务放大，不能称为硬件实测 HBM 写量。

“取”只计这批请求交接时，D 从 P 拉取的缓存张量字节；包含选定 D TP 副本、完整传输块和实际传输的 conv 扩展槽。不累计后续 Decode attention 的 KV 读取，不把同一次传输的 P 读与 D 写加两次。单请求结果再乘 16 得到单 P DP 批；再乘 **P DP** 得到整个 P 组这一批请求的总量。DS/GLM 的 D DP 更大，不应将这些同一批请求再乘 D DP。

本次统一只计目标模型主干层及对应 KV；不包含 draft/MTP 网络、vision、共享专家 TP 通信或 attention TP 通信。D Step 的 128 是可编辑的有效 token 数，并非从 speculative 配置强制推导。若分析完整 speculative pipeline，需另加 proposer 的实际调用、缓存和填充。

## KV 的实现依据

源码基线沿用归档 vLLM `c8438a3d40168ce1d9eade0dc15ccbe5d27adb68` 和 VA `842b030f8375e630eb639e0560eac7735d04f700`。另外对照了用户指定仓库 main 的 `1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c`。该 main 的相关 connector 仍采用下列计数语义；其 aligned Mamba 分支新增了状态块规范化，本次无前缀命中的非 aligned 默认不依赖该分支。不能据此推断两个版本全部实现等价。

### Mooncake 传输规则

1. [prompt 裁剪及尾块选择](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1807)：压缩或有状态的模型在 P 只处理 `N=S-1`，D 重算最后一个 token；普通 GLM 无此 state 截断，`N=S`。非状态组按 prompt 长度裁块，压缩组一个块覆盖 `ratio × block_size` 原始 token；SWA 保留至多 `ceil(window/block)+1` 个尾块，并排除占位块 0。
2. [Mamba 状态选择](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L915)：非 aligned 模式选 `len(remote_blocks)-num_speculative_tokens-1` 对应的一个状态块，不传全部历史状态块，也不将 recurrent state 乘 8。
3. [conv/SSM 传输长度](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1190)：同 TP 情况传完整 conv 和 SSM 张量 view；conv view 中包含 speculative 扩展长度。这里 P/D 设置相同，未建模不同 speculative 长度的兼容性。
4. 基线 [注册张量的 block_len 与 stride](https://github.com/vllm-project/vllm-ascend/blob/842b030f8375e630eb639e0560eac7735d04f700/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L2342) 是不同字段。实际长度按 `element_size × prod(block_shape)`；不能把共享 cache 页的 `page_size_padded` 一概当传输字节。
5. [单 KV head 的 source replica 选择](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L3476) 及基线的 attention group 路由选择，使 MLA 的 P TP8 副本不必全部被 TP1 的 D 拉取。Kimi 普通 D TP8 仍需八份 MLA，Qwen 四个 KV heads 在 TP8 有两倍复制。

通用逐组件公式（层数 `L`，每行字节 `b`，P/D 副本数 `p/d`）：

```text
生成字节 = L × b × generated_rows × p
保留字节 = L × b × retained_rows × p
拉取字节 = L × b × transfer_rows × d
完整上下文拉取行 = ceil(N / (ratio × B)) × B
SWA 拉取行上界 = min(ceil(N/B), ceil(W/B)+1) × B
```

SWA 行数是 connector 尾块规则的保守值：实际占位块和 block IDs 可使其更少。默认 `N=262143` 的非边界尾部与此规则吻合；其它长度仍按上界展示，不能把这个估算冒充实际 `length_list`。有 prefix hit 时应以未命中的实际块列表重算。

### DeepSeek V4 10T / Pro

两个归档 config 在本次相关维度相同：61 层，30 个 C4、31 个 C128，head_dim=512，Indexer dim=128，SWA=128；因此结果相同。“10T”标签不证明整个 checkpoint 的张量组成与 Pro 相同。

[主缓存布局](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/layer/attention/layer.py#L31)、[compressor state](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/deepseek_v4/compressor.py#L154) 和 [Indexer](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/deepseek_v4/indexer.py#L94) 支持如下 A3 计数。P DSA CP 路径先获得完整 cache 输入，再更新各 TP 副本，见基线 [full-sequence cache updates](https://github.com/vllm-project/vllm-ascend/blob/842b030f8375e630eb639e0560eac7735d04f700/vllm_ascend/attention/context_parallel/dsa_cp.py#L1402)。

| 组件 | 每行 B | 生成/保留行（每层） | 默认拉取行 |
|---|---:|---|---:|
| SWA KV | 512×2 | N / 128 | 256 |
| C4 KV | 512×2 | floor(N/4) / 同左 | 65,536 |
| C128 KV | 512×2 | floor(N/128) / 同左 | 2,048 |
| C4 Indexer | 128×1 + 2 | floor(N/4) / 同左 | 65,536 |
| C4 KV/score state | 4×512×4 | 最后 8 行 | 16 |
| C128 KV/score state | 2×512×4 | 最后 128 行 | 160 |
| C4 Indexer state | 4×128×4 | 最后 8 行 | 16 |

P 乘 8，D 乘 1。短输入的有效 state 行取 `min(N,window)`。A5 的压缩字节布局、scale 宽度以及 BF16 cache 分支不同，不适用这里的 A3 行宽。

### GLM5.3

使用 78 个主干 MLA 层，21 个拥有 Indexer 的层。每个 token 分别为 `(512+64)×2` 和 `128×2` B；P 八份，D 一份。[GLM Indexer 选择](https://github.com/vllm-project/vllm/blob/c8438a3d40168ce1d9eade0dc15ccbe5d27adb68/vllm/model_executor/models/deepseek_v2.py#L1118) 按层索引/频率跳过 Indexer，不能将所有 78 层都加一份 Indexer。

旧案例把一个 MTP 层及其 Indexer 算进了 GLM，得到 377.5 GiB/16 请求；这次统一主干范围为 **372 GiB**。这是范围修正，不是声称压缩或传输优化带来了 5.5 GiB 节省。

### Kimi K3 / Qwen3.8 2.4T

Kimi 的 24 层 MLA 每 token 为 `(512+64)×2` B，每个普通 TP Rank 一份；69 层 KDA 的 conv 总宽度 `3×96×128`，SSM 为 `96×128×128` 个 FP32 元素，按 TP 分片。

Qwen 的 23 层 GQA 每个 TP Rank `2×max(1,4/8)×256×2` B/token；69 层 GDN conv 总宽度 `2×16×128+128×128`，SSM 为 `128×128×128` 个 FP32 元素，按 TP 分片。[GDN/KDA state shape](https://github.com/vllm-project/vllm/blob/c8438a3d40168ce1d9eade0dc15ccbe5d27adb68/vllm/model_executor/layers/mamba/mamba_utils.py#L256) 和 [基线 Kimi conv 扩展修正](https://github.com/vllm-project/vllm-ascend/blob/842b030f8375e630eb639e0560eac7735d04f700/vllm_ascend/ops/kimi_kda_state.py#L22) 给出具体维度。

两者的有效 conv 尾部均为 `kernel_size-1=3` 行，传输 view 为 `3+7=10` 行；SSM 只传 1 个最终状态。N=S-1 生成的 full-attention cache，按 block 拉取时向上取整至 S。这解释了它们的“取”略大于“存”。

## EP 通信：从 HCCL count 到 MC2 打包

HCCL [AlltoAllV API](https://gitcode.com/cann/hccl/blob/170ddeec539b4d693028ce6e0cf5c58933e4d46d/docs/zh/api_ref/comm_op_interface/HcclAlltoAllV.md) 的每个 `sendCounts[j]` 都是元素个数。对一组通信，全组跨 Rank 发送为 `Σ_i Σ_(j≠i) count[i,j] × dtype_bytes`。自拷贝和 displacements 地址空洞不计。

设 EP 为 `P`，每 Rank 有效输入 `T`，topK 为 `K`，MoE 主干层数 `L`，hidden 为 `H`。均衡情况下，每层每个源 Rank 发给一个目的 Rank 的期望路由实例为 `T×K/P`。因此一次 Step 所有层跨 Rank 的总实例数：

```text
R = L × T × K × (P-1)
```

负载均衡只保证此端点估算，不能推出物理链路跳数或每条链路平均负载。若单条边的期望 token 数不是整数，这是均衡/期望模型，不伪造实际整数 split matrix。EP 组已经覆盖 DP，不再乘 DP。

### Prefill：AllToAllV

[VA dispatcher](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/ops/fused_moe/token_dispatcher.py#L437) 包含 token dispatch、反向 combine，以及量化时单独的 scale AllToAllV；还会 AllGather 各 Rank 的 per-expert histogram。

```text
dispatch payload = R × (H × dispatch_dtype_bytes + scale_bytes_per_token)
combine payload  = R × H × combine_dtype_bytes
histogram AG     = L × E_physical × histogram_dtype_bytes × P × (P-1)
```

第三项采用 HCCL [Ring AllGather](https://gitcode.com/cann/hccl/blob/170ddeec539b4d693028ce6e0cf5c58933e4d46d/docs/zh/user_guide/coll_algo_intro/Ring.md) 的显式算法模型。histogram dtype 默认 **4 B 假设**，应以实际 `torch.histc` 输出 dtype 校准；Excel 可编辑。主表同时列三项，不隐藏计数交换，也不将 Ring 假设包装成 runtime 选路事实。LoRA、额外路由控制等未计。

### Decode：A3 MC2

HCCL 的 [MC2 接口](https://gitcode.com/cann/hccl/blob/170ddeec539b4d693028ce6e0cf5c58933e4d46d/include/hccl_mc2.h) 定义 count、dtype、通信引擎和资源上下文，但不包含全部 MoeDistribute token 打包规则。为完成用户要求，另取 CANN **ops-transformer `f48a9346d7b638ed2854bb98eafeb6e825e75cff`** 的对应 kernel；版本是分析取证固定点，未验证与部署 CANN 的二进制配套。

[DispatchV2](https://gitcode.com/cann/ops-transformer/blob/f48a9346d7b638ed2854bb98eafeb6e825e75cff/mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV2.md) 本身是 EP AllToAllV 语义。VA 调用中的 MC2 `tp_world_size=1` 是算子内部 TP 参数，不是整个模型 TP；Kimi/Qwen 的模型 TP8 不应因此改成 TP1。

所选 A3 [FullMesh 分支](https://gitcode.com/cann/ops-transformer/blob/f48a9346d7b638ed2854bb98eafeb6e825e75cff/mc2/moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_dispatch_v2_full_mesh.h#L369) 使用 480 B 数据区 + 32 B flag 的 512 B 分块；[量化初始化](https://gitcode.com/cann/ops-transformer/blob/f48a9346d7b638ed2854bb98eafeb6e825e75cff/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_quant.h#L63) 决定 scale 和对齐位置。记 `align32(x)=ceil(x/32)×32`：

```text
dispatch_record = ceil((align32(align32(H×d)+scale)+32)/480) × 512
dispatch 数据写入 = R × dispatch_record
dispatch 计数写入 = L × E_physical × (P-1) × 32
combine 数据写入  = R × H × 2
combine 标志写入  = R × 32
```

dispatch 每个源 Rank 向每个专家写一个 32 B count/flag 槽，见 [SendStatus](https://gitcode.com/cann/ops-transformer/blob/f48a9346d7b638ed2854bb98eafeb6e825e75cff/mc2/moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_dispatch_v2_full_mesh.h#L1014)。combine 每个返回实例另写 32 B 标志，数据部分按 H×dtype 复制，见 [CombineV2](https://gitcode.com/cann/ops-transformer/blob/f48a9346d7b638ed2854bb98eafeb6e825e75cff/mc2/moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2.h#L830)。排除本 Rank 自写后将四项相加，主表叫 **MC2 已建模远端写入量**。

BF16 的 H=7168/6144/8192 对应 dispatch record 为 15360/13312/17920 B。这与只计 `H×2` 的旧 hidden-payload 模型有可解释的差额。

同一源码还有普通 DispatchV2 和 hierarchy 分支。普通分支的有效记录是 `align32(align32(H×d)+scale)+12` B，不应直接套 FullMesh 512/480 比率；Excel 另列普通记录用于辨别，主表不自动切换。轮询远端读、控制同步的其它事务、重试、HCCS/RoCE/UB 协议和多跳转发未全覆盖，留作未知量。没有用一个未经取证的“MC2 倍率”把 payload 变成物理线速量。

### EP256 的部署约束

DispatchV2 要求 `moeExpertNum % (epWorldSize-sharedExpertRankNum) == 0`；VA 本路径 `shared_expert_rank_num=0`。DS V4 的 384、Kimi 的 896 都不能被 256 整除。主表保留 **条件容量规格**，以 512 / 1024 个物理专家槽计控制开销，相当于各补 128 槽，并假设重映射后路由均衡。实际需 EPLB/冗余专家与路由映射支持；不是填充空 tensor 就完成部署。本次没有改模型配置或声称这些规格启动成功。

EP32 的上述模型以及 GLM256、Qwen512 的 EP256 通过专家数整除检查；这也只是一项必要条件，不是完整运行兼容性结论。

## 复算和验证

```sh
node scripts/export-analysis.mjs kv-ep
npm test
node scripts/build-kv-ep-workbook.mjs outputs/kv-ep
python3 scripts/verify-kv-ep.py
```

计算器和测试不依赖 NPU、网络或 Excel 包；Excel 作者工具的安装方法见 [REPRODUCING](REPRODUCING.md)。原生 Excel 黄色输入直接驱动公式，不需要导入 JSON。

验证包括 HCCL AllToAllV 矩阵对照、MC2 两个分支记录差别、十档工作量缩放、EP256 条件约束、N-1 和 block 边界、单 recurrent state。构建时对十档 Excel 关键值逐一比对 BigInt CLI，并修改 prompt、请求数、spec 槽、P token 数和 INT8 dispatch，确认公式重算；检查公式错误并渲染全部三个工作表。独立读取保存的 XLSX XML 再比对关键缓存值。

未运行 NPU、HCCL 集群、Mooncake 端到端传输或 Microsoft Excel 应用。生成量与传输量是上述边界内的源码估算；上线验收应提供实际 block IDs / block lengths、通信 dtype / split counts、MC2 tiling key 和 profiling。
