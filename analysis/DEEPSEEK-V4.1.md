# DeepSeek V4.1 Flash：官方配置与 VA main 的实现差异

采集日：2026-09-12。官方 revision：`dba1be0a40aa45a94ad051997016db3960a90277`；VA revision：`1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c`（main 当时最新的 DSA CP 提交）。[文件和哈希](../models/deepseek-v4.1-flash/sources.json)记录了依据。

本次完成配置、全部权重头部、Python 调用路径和数学模型核验；没有执行 NPU 服务、算子精度或 HCCL Profiling。以下“执行”描述的是代码路径，不表示已完成设备验证。

后续的 [EP32 / EP256 KV 存取扩展](KV-EP-SPECS.md#deepseek-v41-flash) 已补齐 Prefill 生成量、阶段末保留量和有条件的 P→D 张量迁移规划，并汇总 DP 域扩大后的整组量。本页原始 `pdTransferBytes=null` 继续表示当前 main 没有已支持的迁移路径；规划数值另存 `plannedPullBytes`。

## 1. 需要单独建模的结构

| 结构 | V4.1 Flash | 分析影响 |
| --- | --- | --- |
| 主体 | 40 层，H=5120，64×512 Attention | 当前 VA 两阶段遍历全部 40 层 |
| MoE | 384 Expert，TopK=6，Shared=1，I=2304 | Target 与 Draft 的专家数不可混用 |
| 压缩层 | 0–1：SWA；2–19：C2；20–39：C1 | 不能沿用 V4 的 C4/C128 公式 |
| KV 源层 | 2、8、14、20 | 38 个长上下文消费者仅保存 4 份长 KV/Index K |
| Index 源层 | 2、8、14、20、24、28、32、36 | 后四个源重新选 TopK，但仍访问层 20 的 K |
| Candidate | 源层 20，2048 个候选块，每块 8 Token，最终 TopK=512 | TopK 输出与 KV 容量不是同一维度 |
| Engram | 层 1/14，两表各约 3.84 亿行，宽 256；每 Token 每表 24 个 Hash ID | 需要节点内查询/返回及 TP 广播通信 |
| DSpark | 3 层，128 Expert，TopK=3，block size=5 | 不能按主体的 384×TopK6 计算 |

层 26 使用层 20 的 KV 和层 24 的 TopK，源层之间通过本进程的共享 buffer/prefix 交接；这不是层间网络传输。见 [layer plan](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/deepseek_v41/model.py#L115)、[Indexer](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/deepseek_v41/indexer.py#L25)。

官方模型卡描述 CED 的 Prefill/Decode 激活参数差异，但 [VA forward](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/deepseek_v41/model.py#L584) 对两个阶段均使用 `for layer in self.layers`。本模型使用 40 次主体层调用，不将 Prefill 直接减半，也不据模型卡推导此实现的吞吐。

## 2. 权重源格式与运行时表示

[官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/config.json)的普通 FP8 使用 **32×32** block scale；Routed Expert FP4 使用每输出行、每 32 个输入值一个 E8M0 scale。Engram Table 使用每行每 32 元素一个 E8M0 scale。不能沿用旧 V4 分析中的 FP8 128×128 scale 数量。

此次通过 48 个 Safetensors 头部核对全部 96,085 个 Tensor，得出 [116 类条目](../models/deepseek-v4.1-flash/weight-metadata.json)。例如 C2 compressor 投影在文件里是 BF16，VA [Compressor 构造](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/deepseek_v41/compressor.py#L60)明确使用 F32。用运行时参数 dtype 替代文件 dtype 会算错下载/存储量，反过来则会算错驻留量。

另一处区别是 Engram Gate：官方 `wkv.weight` 是 FP8 并带 scale；VA 的 [`nn.Linear`](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/deepseek_v41/model.py#L365) 和 [loader](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/deepseek_v41/model.py#L695)要求 BF16。DSpark `main_proj` 也有官方 FP8 与 VA BF16 的差别。原始 HF 文件不是已验证可直接启动的 ModelSlim 制品。

## 3. Attention：DSA CP 叠加的实际事件

令 `T` 为一个 DP 副本在当前 target forward 的 Token 数，`p=TP`，`t=ceil(T/p)`，`Tpad=t×p`，`H=5120`，`A=2 B`（BF16 激活），`L=40`。以下是**每个 TP 组的发送总字节**，采用 [HCCL Ring 基准](HCCL.md)或 AlltoAll 端点 payload，不是物理链路字节。

| 路径 | 本 Rank 输入 | 每组字节 / target Step |
| --- | --- | --- |
| SP：Attention 输入 AllGather | `t×H×A` | `L×t×H×A×p×(p−1)` |
| DSA CP：Token/Head AlltoAll | `t×64×512×A` | `L×t×64×512×A×(p−1)` |
| SP：O Proj 后 ReduceScatter | `Tpad×H×A` | `L×Tpad×H×A×(p−1)` |
| 不启用 SP：O Proj AllReduce | `T×H×A` | `2×L×T×H×A×(p−1)` |
| SP：最终 Hidden AllGather | `t×H×A` | `t×H×A×p×(p−1)` |

依据：[逐层 SP 调用](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/deepseek_v41/model.py#L416)、[CP 全局 KV / 局部 Q](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/attention/context_parallel/dsa_v41_cp.py#L199)、[restore_tp_heads](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/attention/context_parallel/dsa_common.py#L9)。SP helper 可能选 custom collective，Ring 是显式对比算法，不能断言实际 HCCL 总是 Ring。

这里有两个相对旧抽象的重要修正：

- **DSA CP 开启后，完整 KV/Index cache 仍在每个 TP Rank 复制**。先用完整 hidden 更新缓存，再按 Token 范围切 Q。CP 降低 Query 工作量，不能将缓存除以 TP。
- **没有独立的 Index TopK AllGather**。Indexer 所有 Head 复制，查询对应本地 Token；源层与消费者共用 TopK buffer。旧 V4/GLM 的 Indexer 通信行不能直接套入。

## 4. MoE：共享专家与 Routed Expert 分开

VA 的 [Shared MLP](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/deepseek_v4/model.py#L210)在 `use_sequence_parallel_moe` 开启时设置 `disable_tp=True`，复制权重并处理本地 Token；因此当前 SP 路径的 Shared Expert 通信是 0。非 SP 的 TP 分片路径需要结合 MoE runner 的最终归约，不能再按 SP 路径计算。本数值案例限制为 SP 或 TP1。

Routed Expert 单独提供均匀路由 **Hidden-only 期望**。假设每个 Expert 实例独立分发、每 Rank `t` Token、`e=EP`、TopK=`k`、Dispatch/Combine 宽度分别为 `Ad/Ac`：

```text
EP 组 Dispatch = L × t × k × (e−1) × H × Ad
EP 组 Combine  = L × t × k × (e−1) × H × Ac
```

不再乘一次 DP；EP=TP×DP 已覆盖全部 Rank。该式不是 MC2 的协议字节，也不包含量化 scale、计数、负载不均衡或后端的填充/聚合策略。[All2AllV dispatcher](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/ops/fused_moe/token_dispatcher.py#L437)量化时会另行交换 dynamic scale；[MC2](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/ops/fused_moe/token_dispatcher.py#L162)的 Dispatch 和 Combine 量化模式也不同。精确数据需抓取选中的 backend、两个方向 dtype、split counts 与 scale 张量，交给公共 AlltoAllV 计算器。

## 5. Engram：节点内 A2AV 加 TP 广播

README 的“Engram disabled”已落后于代码：[AscendConfig](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/ascend_config.py#L388)默认 `enable_engram=True`，模型构造了 `NodeShardedEngram`。

[查询组](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/deepseek_v41/engram_hbm.py#L59)按主机划分 EP Rank；要求每节点 Rank 数相同、整个 TP 组留在一个节点，并且启用 EP。两表在每个节点各保存一份，按节点内 Rank 连续分行；不能用全局 EP 大小去除单节点内存。

每次 target forward 的路由顺序为：

1. TP leader 提交当前 DP 副本的 Hash ID，其他 TP Rank 不重复提交；所有 Owner（含空闲 DP）参与通信。每 Token 共 `2×(4−1)×8=48` 次查询，代码按 Owner 分组，没有去重。
2. 两表的计数/invalid flag 合并成每 Rank `2×(n+1)` 个 INT64，做一次节点内 AllGather；HCCL 后端默认设备元数据，Gloo/MPI 使用 CPU 路径。
3. 每张表一次 INT64 ID AlltoAllV，Owner 读取本地行后反向 AlltoAllV 返回结果。对角的本地处理不计跨 Rank 流量。
4. 两表结果合并，TP leader 广播给同 TP 组其他 Rank。

依据：[两表合并 route_many](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/models/deepseek_v41/engram_hbm.py#L444)。单查询默认 ID 为 **8 B**，响应为 **256×2=512 B**。即使表存 INT8、FP8 或 MXFP8，默认返回仍是 BF16。内部 `compressed_int8_wire` 默认 False；若显式启用该实验开关，则返回 `256+8×4=288 B`，不能把这个数当默认值。

若有单表计数矩阵 `C[src][owner]`，查询字节=`8×sum(C[src][owner],src≠owner)`，响应计数为 `C` 的转置。新增 `v41EngramRouting` 保留每 Rank 的发送/接收不均衡。只知道 T 时，例子采用均匀 Owner 的 `(n−1)/n` 期望，不把平均值当 leader 或最忙 Owner 的负载。

| 表存储 | 每行常驻字节 | 位置 | 默认响应 |
| --- | ---: | --- | ---: |
| BF16 | 512 | 设备 | 512 B |
| INT8 + group32 F32 scale | 288 | 设备 | 512 B |
| FP8/MXFP8 + group32 E8M0 scale | 264 | CPU | 512 B |

FP8/MXFP8 路径包含 CPU 解码和 BF16 H2D staging；这些不属于节点网络字节，但也不能从性能分析中忽略。表的 source shape、ceil 分行和 gate/临时 buffer 需要另计。

## 6. 缓存：逻辑保留载荷、物理页和 P→D 各自计算

当前 [cache contract](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/core/deepseek_v41.py#L313)仅允许 BF16/auto；源层 Index K 为 INT8 加每行一个 FP16 scale。对一个长度 S 的请求、一个 TP 副本：

```text
长缓存行数 = 3×floor(S/2) + S
长 KV 字节 = 长缓存行数 × 512 × 2
Index 字节 = 长缓存行数 × (128 + 2)
SWA 保留载荷 = 40 × min(S,128) × 512 × 2
C2 ring 容量 = 3 × 32 × 1024 × 4
```

S=262144 时，长 KV 为 640 MiB，Index 81.25 MiB，SWA 5 MiB，Ring 0.375 MiB，共 726.625 MiB/请求。16 请求为 **11.353515625 GiB/TP 副本**；TP8 全部副本为 90.828125 GiB。只有 KV 源层计入长缓存；不能按 38 个消费者再乘。这里按完整 32 行 Ring 保留容量计数，SWA 是逻辑窗口量，均不等于 allocator 总预留。

官方紧凑 FP4 主 KV 每源 Token `(512/2+512/16)=288 B`，加 FP4 Index K/group32 scale 的 68 B 后按 2.5 个源 Token 份额为 890 B/token；VA 此路径的长 KV+INT8 Index 则为 `2.5×(1024+130)=2885 B/token`。这是紧凑格式的对比；[官方 Python 参考实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py)通过原位量化模拟数值，cache buffer 仍是浮点 Tensor，不能把 890 当成该参考实现的实际分配字节。当前工具采用 VA BF16/INT8 口径，不将官方 890 直接套入。

[物理 allocator](https://github.com/GDzhu01/vllm-ascend-v41-private/blob/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c/vllm_ascend/core/deepseek_v41.py#L137)的 4 个 slot 每全局 ID 为 `[131072,131072,131072,147712]` 字节，总计 **540928 B**。无 Draft 时为 12 个组/51 份资源；DSpark 加 1 个组/3 份 SWA 资源，仍为 4 个 slot。总预留为 `N_global_ID×540928`，N 包括 null/free ID 和各组 live ID。slot 共享 backing，不共享同一 live 数据；不能把该总数再乘组数，也不能以逻辑 Token 字节估算精确 allocator 配额。

此 main 的源码/README 不构成可用 V4.1 P→D 路径的端到端验证。复算器的 `pdTransferBytes=null`；不把缓存容量填成已实现的 P→D 通信量。Prefix cache、Ring 私有状态、Engram n-gram history、Draft SWA 的迁移约定还需要独立验证。

## 7. 对比案例与复算

Prefill：T=16384，TP8×DP4 / EP32，SP=1、DSA CP=1；Decode：T=128，TP1×DP32 / EP32，SP=0、DSA CP=0。每节点 8 Rank，两个阶段都按 40 层；投机默认关闭。每组发送量如下：

| 事件 / 通信域 | Prefill MiB/Step | Decode MiB/Step |
| --- | ---: | ---: |
| Attention 输入 SP / TP | 44,800 | 0 |
| CP Head AlltoAll / TP | 35,840 | 0 |
| Attention 输出 RS / TP | 44,800 | 0 |
| 最终 Hidden AG / TP | 1,120 | 0 |
| MoE Dispatch，仅 Hidden 期望 / EP | 148,800 | 9,300 |
| MoE Combine，仅 Hidden 期望 / EP | 148,800 | 9,300 |
| Engram ID 期望 / 节点 | 5.25 | 0.328125 |
| Engram BF16 响应期望 / 节点 | 336 | 21 |
| Engram 合并广播 Tree 基准 / TP | 2,688 | 0 |

节点 Metadata Ring 基准另为 0.0076904296875 MiB/Step。**通信域不同，不能把此列直接合计成集群总量**。需要集群量时，TP 项按 DP 组数扩展，节点项按节点数扩展，EP 项仅计一次。

```bash
node scripts/export-analysis.mjs communication deepseek-v4.1-flash
node scripts/export-analysis.mjs engram examples/hccl/engram.json
npm test
```

当前事件清单未量化 Embedding/Logits、DSpark draft forward/aux hidden handoff、Vision 路径、MoE counts/scales 及后台同步。开启投机后 target Token 增加不等于所有 draft 成本已包含；VA 仅允许 DSpark，配置需 1..31 speculative tokens，Ring 不能保留 S=32 的最坏拒绝历史。PP/PCP/DCP 必须为 1；DSA CP 是 TP Token 切分，不能视作 DCP/PCP 已支持。跨阶段假定拓扑用于比较，没有进行部署性能推荐。
