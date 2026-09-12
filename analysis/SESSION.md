# 模型分析记录：权重、通信与 P/D 场景

分析日期：2026-09-04。归档及 HCCL 校正日期：2026-09-12。

本记录保留最终结论与配置，使读者无需访问原始会话即可复算。早期已撤回的方案不作为当前方法。

## 最终范围

长期方向是模型性能仿真，包括计算量、访存量、通信、MFU/MBU 及 Profiling 校准。本次落地范围是权重、权重切分、按并行策略拆分的通信量，另有一个固定 256K 场景的 P→D 缓存传输案例。

权重由模型配置和代码的 Tensor 构造推导，不要求 Safetensors。`q_a_proj`、Norm、量化数据与 scale 各自列行、各自设置字节宽度。不能把整个模块统一乘一个主权重 dtype。

每个模型单独一个目录和一份 Excel，Prefill / Decode 独立页签，参数在表内直接可编辑。config 留在目录内，来源证据在 Markdown 中。最终参考布局保留浅色表头和输入单元格，没有额外封面或总结页。

## 模型差异

| 模型 | 本次建模重点 | 产物 |
| --- | --- | --- |
| Kimi K3 | KDA/MLA 混合；专家与共享专家；MLA cache 与 KDA 状态 | 权重/通信 Excel；256K 案例 |
| Kimi K3 DSpark | 独立 draft 模型、输入投影和预测层 | 权重/通信 Excel |
| DeepSeek V4 Flash / Pro | DSA、压缩、Indexer、HC、MoE；DSA CP 下部分权重复制 | 两份权重/通信 Excel；Pro 的 256K 案例 |
| DeepSeek V4 10T | 专用配置，不能用 Flash 代替 | config；256K 案例 |
| GLM 5.3 | MLA/Indexer、DSA CP；索引缓存副本 | 权重/通信 Excel；256K 案例 |
| Qwen 3.8 2.4T | Full Attention/Gated DeltaNet、MoE 及 MTP | 权重/通信 Excel；256K 案例 |
| Qwen 3.8 27B | Dense 模型，EP 为零；Full Attention/GDN 状态 | config；256K 案例 |

名称沿用会话和配置来源，不代表已实测支持的硬件/软件组合。当前 DSA CP 事件抽象尚未完成 DeepSeek 与 GLM 每条执行路径的一一对照，不能因共用 Collective 就认为两个模型调用位置相同。

## 两类产物的参数不同

逐模型工作簿是较早的可编辑分析：默认 TP8、DP1、SP 开；EP 和投机步数依模型而定；共享专家默认 TP8。PCP / DCP 默认 1（关闭），Prefill 才有 PCP，非 DSA 模型 Decode 才有普通 DCP。DeepSeek/GLM 表内有独立 DSA CP 开关。

后期通信需求表是另一个固定场景，采用下列最终设置，不能用逐模型表默认值替代：

| 模型组 | Prefill | Decode | 共享专家 |
| --- | --- | --- | --- |
| DeepSeek V4 10T / Pro、GLM 5.3 | TP8 映射 DSA CP8、SP 开、DP4、EP32 | TP1、DP32、EP32；SP / DSA CP 关 | DP 复制 |
| Kimi K3 | TP8、SP 开、DP4、EP32 | TP8、SP 开、DP4、EP32 | TP8 |
| Qwen 3.8 2.4T / 27B | TP8、SP 开、DP4、EP32 | TP8、SP 开、DP4、EP32 | 2.4T 为 DP；27B 无 MoE |

输入为单请求 262,144 token、单 DP 16 并发、`max_num_batched_tokens=16,384`、投机 7 步。Prefill 为 16,384 token/Step，Decode 为 16 × (7 + 1) = 128 token/Step。投机验证行数不等于最终接受的输出 token 数。

TP/SP/DSA CP 按一个 8 Rank 组统计；EP32 组包含所有 32 Rank，不能再额外乘 DP。DeepSeek/GLM 从 TP8→TP1、DP4→DP32 后，每 Rank 的 Decode EP 输入由 16 token 变为 128 token。

## 通信量与 P→D

- 每 Step 按指定通信组汇总发送端 Tensor 字节。一次传输只在发送端计数，接收端不再加一次。
- MoE dispatch / combine 与共享专家按不同调用/组分别列出。EP 的 `31/32` 是均匀路由期望，不是实测 split vector。
- Attention 行是 SP 或 DSA CP 的模块视图，与策略行重叠，不能把需求表的 15 行求总和。
- P→D 按一个源 DP 的 16 个完整请求计算一次迁移，包含目的端所需 cache 副本及本案例计入的状态缓冲。不按 Step，不再乘集群 DP32。
- GLM Decode TP1/DCP1 的 Indexer 只计一份；Kimi TP8 的 MLA cache 按 8 份复制；Qwen 考虑 KV head 不足 TP 时的复制。
- DeepSeek 的压缩 KV、滑窗、Indexer 和压缩器状态分开；Kimi KDA / Qwen GDN 的卷积、递归状态另计。这些是模型布局假设，不由 HCCL 定义。

通信能力矩阵保持空白。共享池、P/D offloading 行的 `0 B` 表示本场景不统计，不代表实际部署永远没有流量。MiniMaxH3、WAN2.2 未完成填写。

## HCCL 校正与未验证事项

旧表的 Ring 系数在原有输入口径下成立。本次校正计数语义和适用范围，详见 [HCCL.md](HCCL.md)。公共模块新增 API count 归一化、AlltoAllV 计数矩阵、自拷贝排除和大整数检查。公共场景函数修正非 SP、TP>1 分支，使 Attention 使用输出 AllReduce；SP padding 改为向上取整。最终 256K 场景没有走到这些错误分支，归档数值保持一致。

仍需运行时证据：每个模型实际 dispatch 宽度/精度、路由元数据/scale、MTP/draft 调用次数、DSA/SFA 分支、KV connector 去重/压缩/块对齐，以及状态缓冲中实际必须迁移的部分。HCCL 不能替代模型执行路径与 KV connector 的核对。

Workspace、激活驻留、KV 容量规划、offload 命中率和吞吐估算不在逐模型工作簿范围。公式联动与历史数值回归不等于 NPU 端到端验证。
