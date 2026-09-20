# PD KV 传输的一分钟场景估算

2026-09-20。六个模型 × EP32/EP256 × TPOT 5/10/20/30 ms × 三组输入长度/TTFT，共 **144 个组合**。在 EP 汇总工作簿以及对应五份已有逐模型工作簿中新增「分钟场景」页；DS V4 10T 在 EP 汇总工作簿中。Flash 0731 和 DSpark 不是本次六模型之一，没有借用其名字替代 10T。

本页替代“所有层一直争用”的场景假设。原表保留为手动覆盖率敏感性分析，其 100% 覆盖率不再代表这里的默认结果。

## 已确认的输入与统计口径

用户确认：每个 P DP 连续工作，每完成一批 prefill 就交接 KV，每批 **16 请求**；将 **16K/4s、256K/20s、1M/60s** 的 TTFT 暂作 P 批次完成间隔。K=1024，M=1024²。示例带宽 **4800 GB/s**，不是 Gb/s。EP、TP 通信与 KV 传输带宽各设独立输入，初始均为 4800。

TTFT 通常包含调度、计算、交接等耗时，不天然等于稳定的批次间隔。这是用户确认的容量场景假设，不从服务端 TTFT 指标直接推出实际到达率。P 耗时变化不会在本页反向修改到达率；这是一个给定输入负载的 D 侧评估。

默认一组 P EP 对一组 D EP；P/D 组数均可编辑。模型内仍采用 P TP8；DS V4 / GLM / V4.1 的 D TP1，Kimi / Qwen 的 D TP8。EP 增大，P/D 的 DP 域一起扩大。各组、各 DP 同步完成批次，默认每周期形成 **一个聚合 KV 窗口**。不是说每周期只调用一次传输 API。

「每周期错峰批数」m 可改为大于 1：把总字节均分为 m 个等间隔窗口，平均 KV 需求量不变。它模拟错峰交接，不是实测调度。默认 m=1 时，三组场景分别产生 **15、3、1 个窗口/分钟**。这是一分钟的稳态速率；不采用“从空系统启动、第一批在 60 秒末完成，因而第一分钟为零次”的口径。

请求交接数和 KV 窗口数单独记录。forward 次数按**一个 D DP 副本**统计，采用该副本各 TP Rank 的均衡通信时长。不能把它再次乘 EP 当作一个请求的暴露次数。

Decode 沿用原规格的 **128 输入 tokens / DP / forward**；它与 P 侧每批 16 请求是独立工作量输入。**平均产出 tokens / forward** 初始为 1，可填投机解码实测接受结果：

```text
forward 步间隔 F = TPOT × 平均产出 tokens / forward
基线 forward 数 N = 60000 / F
```

不能仅因评估了 8 个候选就断言一次 forward 产出 8 个 token。输入 token 数和实际产出量都可以按模型、EP 改。EP＋TP 带宽时间已超过给定步间隔时显示「TPOT 不自洽」，并列出扣除当前 TP 带宽时间后，剩余时间容纳 EP 所需的最低带宽；通过该检查不代表算力或 HBM 已满足 TPOT。

## KV 窗口怎么得到

使用 [KV 源码模型](KV-EP-SPECS.md)中的 **P→D 取量** V，不用 Prefill 生成量代替。V 是一个请求在全部 D TP Rank 上的张量字节；共享、压缩、尾块、Indexer、recurrent/conv state 按各模型实现计算。三个长度各自重算，不按长度简单倍乘最终结果。工作簿组件区同时保留生成、保留和取量。

设 P/D EP 组数为 Gp/Gd、EP 为 E、P TP 为 Tp、D TP 为 Td、每 P DP 批请求数为 R、批次周期为 C ms、每周期错峰批数为 m：

```text
P DP / 组 = E/Tp                           D DP / 组 = E/Td
全系统交接请求/min = Gp × (E/Tp) × R × 60000/C
每 D DP 交接请求/min = 上式 / (Gd × E/Td)
P 每 Rank 每窗口发送量 VP = R × V / Tp / m
D 每 Rank 每窗口接收量 VD = VP × Gp/Gd
窗口周期 P = C/m
窗口时长 W = MAX(VP, VD) / (Bkv × 10^6)     ms
KV 负载率 u = W/P
```

假设 P/D 端点按 Rank 均衡分摊，P/D 两端同口径的有效 KV 带宽均为 Bkv，以较慢端限定窗口。若实际源 Rank 聚集、跨服务器瓶颈或 P/D 带宽不同，应以实际瓶颈调整输入或扩展映射，不能把总链路标称速率作为每 Rank 有效带宽。

u≥1 时，KV 传输无法在下批前排完，表中显示「KV 超载」，不再给稳定 TPOT 估算。没有把多个重叠窗口累加为大于 100% 的覆盖率，也没有忽略积压继续给一个看似稳定的数字。

同 P/D 配比下，EP32→EP256 使全组交接请求数变成 8 倍；**每 Rank KV 窗口不自动变成 8 倍**。EP 通信字节仍按新 EP 重新计算，4800 GB/s 也可按 EP 分别修改。

## 哪些层与窗口重叠

目前没有逐层 profiling。本页采用**相位平均估算**：L 个等效层段均匀分布在 F 内，L 沿用 MoE 层数。每层段由相邻的 EP、TP 两段组成。无竞争 EP 时间 tE 来自 [A3 FullMesh MC2 公式](KV-EP-SPECS.md#decodea3-mc2)，TP 时间 tT 按下节 HCCL Ring 字节除以独立 TP 带宽得到。合并段长 c=(tE+tT)/L，内部间隔 g=(F−tE−tT)/L。

这是用于当前无 trace 场景的等效排布：Dispatch/Combine 合并；Attention、Dense、共享专家和首尾 TP 字节均摊到 L 段。它保留总字节与 EP/TP 独立带宽，不复现真实算子位置，也不推断真实的层序或跨流并发。缺少逐层时序时，不能把这些层段次数解释为具体某个算子或层 ID。

KV 窗口与 forward 的相位在一个到达周期内均匀分布。对每个组合记录：

- **重叠 FW**：至少一个 EP 或 TP 段与 KV 窗口相交，两者取并集，不把 FW 次数相加。
- **全通信 FW**：该步所有 EP＋TP 段都落在 KV 窗口里；不表示整步计算也全部在窗口里。
- **部分通信 FW**：重叠 FW 减全通信 FW。可能只影响一层，也可能每层都只影响一部分。
- **重叠层段**：与 KV 窗口相交的等效合并层段数；不是不同层 ID 的个数，也不是 MC2 底层消息数。
- **等效全通信 FW**：按受影响通信持续量折算的工作量，在明细中单列，不冒充实际发生的 forward 次数。

令 span=(L−1)F/L+c。在 0<W<P、F≤P 下，对相位均匀的 forward：

```text
A(W) = 1 − [(L−1) MAX(g−W, 0) + MAX(P−span−W, 0)]/P
重叠 FW/min    = N × A(W)
全通信 FW/min  = N × [1−A(P−W)]
部分通信 FW/min = 两者之差
重叠层段/min   = N × L × MIN(1, (W+c)/P)
等效全通信/min = N × W/P
```

零 KV 或零通信另走零分支。这里从**通信段与窗口的交集**计算次数，不把 `W/F` 直接当受影响 forward 数，也不把“12/60 层”无条件当 20% 的通信量。小数是相位平均期望；实际单个一分钟窗口的整数次数取决于相位。

## TP/SP 字节与 EP/DP 的关系

修正前，分钟页的 TP8 只参与 DP 域和 KV 映射，增时仅使用 EP 字节，遗漏了 K3、Qwen 的 TP/SP 带宽项。本次补入六项互斥的通信量：Embedding、Attention SP、Dense MLP、共享专家、Final Hidden、Sampling。SP 中的 Attention 不再额外重复求和。原逐模型 Prefill/Decode 事件表已有 TP，这次修正的是分钟页。

设 D TP=p、每 DP 输入 tokens=T、n=ceil(T/p)、hidden=H，BF16 为 2 B。按 [HCCL Ring 口径](HCCL.md)，**每 Rank 发送字节**为：

| 事件 | 字节/Rank/步 |
|---|---|
| Embedding AllReduce | T×H×2×2(p−1)/p |
| 每 Attention 层 AG＋RS | 2×n×H×2×(p−1) |
| 每 Dense 层 AllReduce | n×H×2×2(p−1)/p |
| 每共享专家 TP 层 AG＋RS | 2×n×H×2×(p−1) |
| Final Hidden AllGather | n×H×2×(p−1) |
| Sampling candidate AllGather | T×256×(2＋4)×(p−1) |

Sampling 沿用原表每输入 token 一行、256 个候选、BF16 值＋INT32 索引的服务假设，不把全 vocab logits 当迁移量。层数和共享专家策略沿用归档配置与源码路径：

- K3：93 层 Attention、1 层 Dense、92 层共享专家 TP。[Kimi MLP 的 TP 投影](https://github.com/vllm-project/vllm-ascend/blob/842b030f8375e630eb639e0560eac7735d04f700/vllm_ascend/models/kimi_k3.py#L601)及 [SP gather/scatter 实现](https://github.com/vllm-project/vllm-ascend/blob/842b030f8375e630eb639e0560eac7735d04f700/vllm_ascend/ops/linear_op.py#L295)。
- Qwen：92 层 Attention（含 GDN）、无 Dense 层，共享专家沿用 SP 下 DP 复制路径，不加 TP。[shared expert 传入 SP 设置](https://github.com/vllm-project/vllm/blob/c8438a3d40168ce1d9eade0dc15ccbe5d27adb68/vllm/model_executor/models/qwen3_next.py#L153)以及 [MLP 的 disable_tp](https://github.com/vllm-project/vllm/blob/c8438a3d40168ce1d9eade0dc15ccbe5d27adb68/vllm/model_executor/models/qwen2_moe.py#L82)。

默认 T=128、p=8，K3 为 **600.6784 MB/Rank/步**，Qwen 为 **344.522752 MB/Rank/步**（十进制 MB）。EP32→EP256 时，p 仍为 8，DP 从 4 变成 32；每 Rank TP 字节不变，整个 EP 域包含的 TP 组数变成 8 倍。整域 TP 总量应乘 EP，不应把单个 TP 组总量误当单 Rank，或再把每 Rank 量乘 DP 带入单步延迟。

DS/GLM/V4.1 的 D TP1 在这些 TP collective 上为零，本次增时结果不变。MC2 的内部 `tp_world_size=1` 不等于 K3/Qwen 的模型 TP1。TP/SP 是 Ring 算法载荷，MC2 是所选分支的远端写入模型；它们各自除以同口径有效带宽，不能把混合字节直接当物理线速计数。

## 窗口结束时恢复带宽

窗口内带宽降幅 d=10%，窗外恢复正常。对于一个长度为 c 的基线通信段，如果它在窗口结束前没有传完，剩余字节按正常速率完成。不能对这个完整通信段都乘 1/0.9。

固定一个通信段的开始相位，在整个 KV 周期内积分其完成时间。令 a=1−d，且 c≤P−W，则平均单段增加时间为：

```text
若 c ≤ aW： e = [d c W/a − d²c²/(2a²)]/P
若 c > aW： e = [d c W + d²W²/2]/P
```

它是一个孤立通信段在固定窗口下的精确相位积分。默认窗口相隔很远，满足 c≤P−W。若修改参数导致不满足，或 F>P，表中显示「需时序仿真」，不继续套这个公式。公共测试使用逐段速率积分独立核对长、短窗口及其边界。

EP 和 TP 分别用 cE=tE/L、cT=tT/L 代入单段公式 e(c)，不用合并段长套同一个带宽。固定基线排程的一阶敏感性模型中，设各自共享比例 s、关键路径暴露率 h：

```text
ΔTE = N × L × e(cE) × sE × hE
ΔTT = N × L × e(cT) × sT × hT
ΔT_60s基线工作量 = ΔTE + ΔTT
平均 TPOT 增量    = TPOT × ΔT / 60000
TPOT 增幅        = ΔT / 60000
```

「60s 工作增时」表示原本耗时一分钟的基线工作量，预计额外增加多少毫秒。**不是实际墙钟一分钟中的调度仿真，也不是一分钟内丢掉的 token 数**。每段变慢后，后续层与下一步的相位可能移动；这里不反馈这种位移，也不反馈 P 到达节奏、D batch 大小或排队。层放置和相位平均假设对短窗口尤为重要。共享/暴露系数只缩放耗时；表里的几何重叠次数仍列出潜在重叠，即使这些增量被计算隐藏。

默认 sE=sT=hE=hT=1 是完全共享、完全暴露的输入场景。如果 TP 链路与 KV 不共享瓶颈，令 sT=0，此时 TP 字节和基线 TP 时间仍保留，TP 竞争增时为零。若 TP/EP 实际并发，两个原始带宽项之和也不是实测 critical path；应以 profiling 校准暴露率，TPOT 不自洽检查仍是本表串行带宽时间的检查。

10% 必须指**KV 活跃窗口内**的降幅；如果实测 10% 已是整段 forward 的平均损失，就不能再用这里的窗口占用二次折减。

## 一个可复算的例子

Kimi K3、EP256、TPOT 10 ms、256K/20s，P/D 各一组、16 请求/P DP、128 D tokens/DP/步、平均产出 1，三个带宽均 4800 GB/s，EP/TP 共享和暴露比例均为 100%：

| 指标 | 相位平均估算 |
|---|---:|
| KV 窗口/分钟 | 3 |
| KV 窗口持续时间 | 24.361 ms |
| 基线 forward/分钟 | 6000 |
| 与 KV 重叠的 forward/分钟 | 10.277 |
| 所有 EP＋TP 段都重叠的 forward/分钟 | 4.340 |
| 只有部分通信段重叠的 forward/分钟 | 5.937 |
| EP 增时/基线分钟 | 0.118494 ms |
| TP 增时/基线分钟 | 0.101620 ms |
| 基线一分钟工作量增加时间 | 0.220113 ms |

相同配置下改成 16K/4s：15 个窗口，每个 1.712 ms，约 17.409 个 forward 部分重叠，全通信重叠为 0；改成 1M/60s：一个 96.839 ms 窗口，约 10.673 个 forward 重叠，其中 8.694 个是全通信重叠。相同 256K/20s、EP256、TPOT10 的 Qwen 总增时从 EP 单项的 **0.073053 ms** 变为 **0.122896 ms**，其中 TP 为 **0.049843 ms**。数值小也与 4800 GB/s 示例带宽很高有关，不能直接视作硬件测量。

## Excel 输入与复算

每份新增「分钟场景」页的位置一致：`N5` 为 KV 带宽、`N7` 为 EP 带宽；`B5` 为每 P DP 请求数，`F5/J5` 为 P/D 组数；`B7/F7` 为 D 输入和平均产出 tokens/步；`B9` 为每周期错峰批数；`J7` 为降幅。`B15:C17` 是长度/TTFT，`F15:F18` 是 TPOT。第 23 行起可按模型/EP 覆写 I/J 列带宽和 K/L 列 token 设置。

TP 修正已应用于总表、K3、Qwen 的分钟页：`F11` 为 TP GB/s，`J11/N11` 为 TP 共享/暴露比例；参数表 P/Q/R 列可逐模型/EP 覆写。结果 Q/R 列列出 TP GB/Rank/步和 TP ms，S/T 列分别列 EP/TP 增时，L 列为两者合计。计算区 Y:AD 是六项 TP 字节，AE:AL 是 TP 时间与合计过程。其余 D TP1 模型文件没有新增无效控制，结果不变。

CLI 的 `allCommunicationInsideStepsPerMinute` 对应新的全通信列；`partialStepsPerMinute` 是 EP＋TP 并集的部分重叠次数。原 `allEPInsideStepsPerMinute` 继续提供 EP 单项分类，不能再与新的部分通信次数相加。`commMs` 仍为 EP，`tpCommMs` 与 `totalCommMs` 分别提供 TP 和两者合计。

新页的输入独立于原表历史默认参数，以免 EP8、投机步数等不同基线混入 EP32/256 对比。原表已填带宽保持原值，修改新页会立即重算新页公式。所有原标签、公式和样式保留。

高带宽示例下单 token 增量很小，因此结果 M 列以 **μs** 显示，N 列增幅以 **ppm** 显示（1 ppm = 0.0001%），避免小数显示为零。L 列的一分钟工作量增时仍是 **ms**；CLI 的 `meanTpotIncreaseMs` 和 `tpotIncreaseFraction` 保持毫秒与无单位比例。

```sh
node scripts/export-analysis.mjs minute
node scripts/export-analysis.mjs minute path/to/options.json
node --test tests/*.test.mjs
python3 scripts/verify-minute.py
```

`options.json` 接受 `MINUTE_DEFAULTS` 的键，以及 `cases` 和 `tpots`。生成 Excel 使用现有 Artifact Tool 环境：

```sh
node scripts/update-minute-workbooks.mjs --inspect
node scripts/update-minute-workbooks.mjs --write
node scripts/update-minute-workbooks.mjs --add-tp outputs/pd-tp
```

`--write` 导入原表并增加分钟页，检测到已有页面会停止。`--add-tp` 定向修正总表、K3、Qwen 的现有分钟页，保留其 KV 公式、输入和其他标签页；两种模式均输出到指定 outputs 目录，不直接覆盖仓库文件。生成后需核对再写回。JSON 示例见 [minute-results.json](../examples/pd-contention/minute-results.json)。

## 来源与范围

模型配置、KV 字节公式、HCCL 与 MC2 字节模型沿用 [KV-EP-SPECS.md](KV-EP-SPECS.md)。普通 Mooncake 后台传输、请求等待和逐层 connector 依据见 [PD-CONTENTION.md](PD-CONTENTION.md#实现依据与重叠范围)及其固定版本源码链接。代码并不保证某次 KV 传输只影响当前请求或固定若干层，因此以外部 KV 窗口与正在执行的 D forward 相交建模。

计算 MoE EP 和上述 TP/SP 带宽项的增量。未计固定通信启动、重试、轮询、Engram、计算或 HBM 竞争；不同链路不共享应调各自共享比例。此页未新增 P 侧 forward 时间仿真，也未将一次完整交接等待加到每个 D step。

384/896 专家在 EP256 仍需 128 个冗余槽与 EPLB 验证。V4.1 的取量继续是固定源码快照的规划值，**没有通过本次表格证明新 main 的 PD 端到端可用性**。1M 等长度是容量推演，不是运行支持声明。
