# PD KV 传输对 forward 耗时的影响

后续已新增 [一分钟窗口场景模型](PD-MINUTE.md)：TPOT 5/10/20/30 ms × 16K/4s、256K/20s、1M/60s，六模型 EP32/256 共 144 档。新页用 KV 字节、交接节奏和有限窗口估算部分 / 全 EP 重叠，默认独立 EP/KV 带宽均为用户指定的 4800 GB/s。本文原有 100% 覆盖结果保留为敏感性参考，不代表新窗口场景。

2026-09-20。假设 KV 传输使共用资源上的 forward **有效通信带宽下降 10%**，则受影响通信的带宽耗时项增加 `1/0.9-1 = 11.111%`。完整 forward 的增幅还取决于通信占比、争用覆盖和计算/通信重叠，不能直接写成 10% 或 11.111%。

已更新七个逐模型 Excel 的 Prefill / Decode，并在 [EP32/EP256 工作簿](../examples/kv-ep-sweep/kv-ep32-ep256.xlsx)追加十二档比较。有效带宽均可直接在 Excel 修改，100 GB/s 只是用户选定的示例输入，不是硬件测量。原权重、通信量、KV 生成/保留/拉取结果和参数均保留。

## Excel 怎么填

逐模型文件在原通信表下方增加「PD 带宽争用耗时评估」：

- **E 列黄色「有效带宽 GB/s」**：逐事件填写无 KV 竞争时的有效带宽。默认通过公式引用该区域的「默认带宽」H 列输入；改 H 可统一更新，直接覆写某行 E 可单独设置 TP、EP、Attention 或 Engram。
- **带宽降幅**默认 10%；**争用覆盖率**、每事件**共享比例**与**暴露系数**默认 100%。独立链路的共享比例设 0；被计算完全隐藏的通信增量，暴露系数设 0。
- **F 列「基线 forward ms」输入**默认空白。填入同配置、同 Step 的无竞争实测耗时后，才显示新的 forward 耗时和百分比。实测基线小于所假设的已暴露通信时间时，显示「基线/暴露不一致」，不能用该百分比。
- **GB/s 是十进制 10⁹ 字节/秒**，不是 Gb/s。若手头单位是 Gbit/s，先除以 8；GiB/s 先乘 `2^30/10^9`。填无竞争带宽；若已经填了争用后的带宽，不能再扣一次 10%。

快速定位各表的输入行。P/D 输入行的 B、D、F、H 分别是降幅、覆盖率、forward 基线、默认带宽；E 列事件带宽在对应区域的明细表内。

| 模型 | Prefill 输入行 | Decode 输入行 |
|---|---:|---:|
| V4.1 Flash | 138 | 138 |
| Kimi K3 | 98 | 97 |
| Kimi K3 DSpark | 73 | 73 |
| V4 Flash 0731 | 138 | 134 |
| V4 Pro 0813 | 138 | 134 |
| GLM5.3 | 85 | 81 |
| Qwen3.8 2.4T | 83 | 82 |

EP 对比工作簿的「规格汇总」第 51 行起是新区域。`B55/D55` 是 P/D 默认带宽；**`M58:N69` 是各模型、各 EP 的 P/D 可编辑带宽**。`F55/H55/J55` 分别控制带宽降幅、覆盖率、暴露系数；`I58:I69/K58:K69` 接受 P/D 实测 forward 基线。未填写基线时，仍能查看毫秒增量，但百分比保持空白。

这里的有效带宽必须与通信字节的计数口径一致。例如每 Rank 端点发送字节应除以同口径的每 Rank 有效带宽，不能直接用双向端口总额定速率，也不能把整个 EP 组的字节除以一张卡的带宽。

## 实现依据与重叠范围

本次检查 upstream VA main `5c80630f28f8529aa82716e58b981a78819ec429`，并复核用户 V4.1 仓库 main `8727bd4e03f614afc1f22df9c53b02b6cf8a09f3`。文件 SHA256、精确链接在 [源码清单](pd-contention-sources.json)。流量公式继续使用此前固定的 HCCL / ops-transformer 分支；没有把新 VA、旧 vLLM 和算子版本声明为已配套验证的运行环境。

| 路径 | 源码行为 | 对耗时评估的含义 |
|---|---|---|
| 普通 MooncakeConnector | [线程池和请求队列](https://github.com/vllm-project/vllm-ascend/blob/5c80630f28f8529aa82716e58b981a78819ec429/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L491)在后台处理接收，[同步 read](https://github.com/vllm-project/vllm-ascend/blob/5c80630f28f8529aa82716e58b981a78819ec429/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1042)发生在后台任务中；普通 connector 的 `wait_for_layer_load` 是空实现 | `sync_read` 不代表把整段 KV 传输串行插入每个正在运行请求的 forward。其它请求可以同时 forward，因此可能竞争同一设备/网络资源 |
| 新到达 D 的请求 | vLLM 固定版本的[调度器](https://github.com/vllm-project/vllm/blob/c8438a3d40168ce1d9eade0dc15ccbe5d27adb68/vllm/v1/core/sched/scheduler.py#L1091)将异步加载请求置为 `WAITING_FOR_REMOTE_KVS`，完成后才使用这些 KV | 该请求自身有交接等待，影响恢复/首 token 延迟；等待不等于每个后续 decode step 都重传完整 KV |
| MooncakeLayerwiseConnector | [cache 写完记录 NPU event](https://github.com/vllm-project/vllm-ascend/blob/5c80630f28f8529aa82716e58b981a78819ec429/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py#L1834)，发送侧 worker 在事件后执行写传输；fallback 路径减少重叠 | P 侧可在前面层的 KV 发送期间继续后续层计算。是否命中早期 hook、是否共用资源影响实际覆盖率。不能因此声称 D 每层都等待：该版本 worker 的 `wait_for_layer_load` 也是空实现 |
| P AllToAllV | [dispatch](https://github.com/vllm-project/vllm-ascend/blob/5c80630f28f8529aa82716e58b981a78819ec429/vllm_ascend/ops/fused_moe/token_dispatcher.py#L512)的 scale/hidden 交换有显式 `wait()`；[combine](https://github.com/vllm-project/vllm-ascend/blob/5c80630f28f8529aa82716e58b981a78819ec429/vllm_ascend/ops/fused_moe/token_dispatcher.py#L566)也等待结果 | dispatch → expert GMM → combine 有依赖，通信变慢可能延长 forward 的关键路径 |
| D MC2 | [调用 DispatchV2](https://github.com/vllm-project/vllm-ascend/blob/5c80630f28f8529aa82716e58b981a78819ec429/vllm_ascend/ops/fused_moe/token_dispatcher.py#L232)和 [CombineV2](https://github.com/vllm-project/vllm-ascend/blob/5c80630f28f8529aa82716e58b981a78819ec429/vllm_ascend/ops/fused_moe/token_dispatcher.py#L331)，[MoE 执行序列](https://github.com/vllm-project/vllm-ascend/blob/5c80630f28f8529aa82716e58b981a78819ec429/vllm_ascend/ops/fused_moe/moe_comm_method.py#L136)记录前后事件；[共享专家多流](https://github.com/vllm-project/vllm-ascend/blob/5c80630f28f8529aa82716e58b981a78819ec429/vllm_ascend/ops/fused_moe/fused_moe.py#L259)提供重叠路径 | 不应给整个融合算子耗时直接乘 `1/0.9`；仅将估算带宽项的增量按暴露系数计入 forward，固定启动、轮询、计算和同步时间需 profiling |

Mooncake 使用独立的 [TransferEngine ascend transport](https://github.com/vllm-project/vllm-ascend/blob/5c80630f28f8529aa82716e58b981a78819ec429/vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py)，并不因为 connector 名称就等价于使用同一个 HCCL communicator。物理 NIC、HCCS、PCIe 或 HBM 是否共享要看部署。这里按用户给出的 10% 有效带宽损失做敏感性分析，不从 Python 调用推断硬件损失。

V4.1 新 main 已包含 `fix(pd): add full_graph_mode for dsa_v41`，而模型 README 仍有未支持迁移的旧说明。本次没有完成端到端 P→D 运行，也没有重新验证新 main 的完整块列表/传输长度，所以保留原 `1933f86` KV 规划快照和 `pullBytes=null`。V4.1 本次结果是**在该 PD 场景成立条件下**的 forward 通信敏感性；不能用旧 README 宣称新 main 一定不可用，也不能仅凭提交名宣称新 main 已通过验证。

## 公式

对阶段内第 i 类通信，`V_i` 是每 Rank 每 Step 的已建模端点字节，`B_i` 是无竞争有效 GB/s，`d` 是降幅，`s_i` 是共享比例，`q` 是争用覆盖率，`e_i` 是暴露系数：

```text
t_i                 = V_i / (B_i × 10^6)                       ms
Δt_comm_i           = t_i × s_i × q × d / (1-d)
t_comm_i_contended  = t_i + Δt_comm_i
ΔT_forward          = Σ(e_i × Δt_comm_i)
T_forward_contended = T_forward_baseline + ΔT_forward
forward 增幅        = ΔT_forward / T_forward_baseline
固定工作量吞吐损失  = ΔT_forward / (T_forward_baseline + ΔT_forward)
```

`q` 定义为按无争用通信工作量计的受影响份额；不是直接拿争用后墙钟时间占比代入。`e_i` 是增量进入关键路径的局部近似，默认 1 表示完全暴露。重叠调度可能随负载改变，极端争用下需重新测量，不能以一个固定系数预测排队或 P99。

如果实测的 10% 已经是**整个阶段平均带宽损失**，保持 `q=1`；只有 10% 指争用活跃窗口内的下降时，才再乘窗口覆盖率，避免二次折减。

例如某阶段无竞争 forward 为 100 ms，受影响且暴露的带宽项合计 30 ms，则增量为 `30/9=3.333 ms`，forward 增幅为 3.333%，固定工作量吞吐下降约 3.226%。这些是公式示例，不是任何模型的实测性能。

## 通信字节的口径

HCCL 的 [count、Ring、AllToAllV 说明](HCCL.md)和 [MC2 已建模分支](KV-EP-SPECS.md#decodea3-mc2)继续适用。令 `P=EP`，`T` 为每 Rank token 数，`L/H/K/E` 为 MoE 层数、hidden 宽度、TopK、逻辑专家数，均衡路由的每 Rank 远端路由数为 `R=L×T×K×(P-1)/P`，物理专家槽为 `E'=ceil(E/P)×P`：

```text
P AllToAllV bytes/rank = R×H×(dispatch_bytes+combine_bytes)
                        + L×E'×4×(P-1)
D dispatch record     = ceil((align32(align32(H×2))+32)/480)×512
D MC2 bytes/rank      = R×(dispatch_record+H×2+32)
                        + L×(E'/P)×(P-1)×32
```

这是默认 BF16、P 4 B 计数 Ring、D A3 FullMesh 分支的端点写量模型。零 tokens 时控制量也按 0。P 动态 scale、其它 count dtype、MC2 ordinary/hierarchy/A5 必须切对应公式，不把此处系数当所有版本的通信库通则。已存在的 EP 汇总表还支持 INT8+FP32 scale dispatch，其耗时区直接链接重算后的字节。

逐模型耗时区沿用原通信事件及当前表参数；P 补直方图控制量，D 替换为 MC2 packed dispatch / combine flags 并补 count 写入。D 当前扩展明确只支持 BF16，输入不符返回 `#N/A`；需 INT8 请用 EP 规格表已有打包选项。其它行直接链接原 K 列每 Rank 每 Step 的 MiB。DSpark 没有 MoE EP，保留它自己的通信事件。它的结果不是 Kimi 主干的第二份 EP。

七模型表还含各自已列的 TP/SP/Attention 等事件；不同历史默认 token、投机步数、EP 和层数口径仍然不同。应使用下面统一的 EP 规格比较 EP32/EP256，不能混用两个表的绝对时间。

## 六模型 EP32 / EP256 示例结果

以下仅为 **MoE EP 带宽项给 forward 带来的毫秒增量/Step/Rank**：100 GB/s、10% 损失、完全共享/覆盖/暴露。P 每 DP 16,384 tokens、TP8；D 每 DP 128 tokens，DS/GLM TP1，Kimi/Qwen TP8。所有数均为均衡负载下的 Rank 平均值，非整个 forward 时长或性能测量。

| 模型 | EP32 P 增量 ms | EP256 P 增量 ms | EP32 D 增量 ms | EP256 D 增量 ms |
|---|---:|---:|---:|---:|
| DS V4 10T | 23.137 | 23.822 | 1.500 | 1.543 |
| DS V4 Pro | 23.137 | 23.822 | 1.500 | 1.543 |
| GLM5.3 | 32.509 | 33.445 | 2.120 | 2.179 |
| Kimi K3 | 93.050 | 95.772 | 0.756 | 0.778 |
| Qwen3.8 2.4T | 66.463 | 68.386 | 0.546 | 0.561 |
| DS V4.1 Flash | 10.837 | 11.164 | 0.713 | 0.733 |

EP 增大同时扩大 DP 域，保持每 DP 工作量时，全组 token 数扩大 8 倍，全组 routed payload 扩大约 8.226 倍。但**每 Rank** payload 只增加约 2.82%，所以在相同每 Rank 有效带宽假设下不能把耗时增量也乘 8。若 EP256 的实际有效带宽更低，直接修改该规格的 M/N 列，会额外放大其耗时。

384/896 专家的 EP256 仍是需要专家冗余槽/EPLB 验证的条件规格。V4.1 表中没有计入 Engram；Engram 在其独立逐模型表内。完整数值见 [results.json](../examples/pd-contention/results.json)，七模型当前参数结果见 [model-results.json](../examples/pd-contention/model-results.json)。

## KV 量怎样影响争用持续时间

同一请求的 KV 交接时间可先写成 `T_KV ≈ V_pull / B_KV + 固定开销 + 排队/转换`。这里应使用「取」的目标副本和实际块量，不使用整个 Prefill 生成量。带宽应对应请求实际使用的传输资源；多 Rank 并发拉取不能把总字节直接除以某个 Rank 的带宽。

沿用 256K、16 请求/P DP 的缓存案例，同一 P DP 批的拉取估算为 DS V4 10T/Pro **35.392 GiB**、GLM **372.000 GiB**、Kimi **871.227 GiB**、Qwen **745.046 GiB**；V4.1 **11.432 GiB 规划值**。在等到达率、等资源映射、等 KV 有效带宽条件下，较大的拉取量可能拉长争用窗口；不能把这些不同拓扑下的批量直接解释为每张卡占网时间。

粗略稳定负载下，某共享链路的 KV 利用率可用 `arrival_rate × bytes_per_arrival_on_link / KV_bandwidth_on_link` 作需求检查，再结合 trace 统计它覆盖了多少 forward 通信。它不是仅凭 cache 大小就能确定的模型常数。当前缺到达率、映射和 profiling，故暴露可编辑 `q`，不从 KV 字节伪造覆盖率。

如果用户给出的 10% 已包含 KV 的影响，就不要再按 KV 大小乘一次 10%。同时应分开记录：新请求的一次交接等待、已有请求每 Step 的 forward 增量、P 侧吞吐变化及排队。这份表计算后两阶段的通信敏感性，不将完整 KV 搬运时间重复加到每个 decode step。

## 复算与验证

公共无依赖计算器是 [lib/pd-contention.mjs](../lib/pd-contention.mjs)，可传自己的 `bandwidthGBps`、`bandwidthLoss`、`sharedFraction`、`contentionCoverage`、`exposedFraction`、`baselineForwardMs`。`loadContentionSweep` 的 `kvOptions` 可改 token 等工作量；Excel 可按事件/模型/EP 使用不同带宽。

```sh
node scripts/export-analysis.mjs contention
npm test
python3 scripts/verify-contention.py
```

可选的 Artifact Tool 更新器 `node scripts/update-contention-workbooks.mjs --write outputs/pd-contention` 读取当前八份工作簿，只重建新增耗时区域并恢复示例输入，输出到指定目录。它不自动覆盖仓库工作簿，也不保留该区域内用户填的实测值；已填写的文件应在 Excel 中直接复算。两个常规生成器也已接入相同扩展。

验证覆盖 14 个阶段扩展、12 个 EP 案例，带宽/降幅/覆盖率/基线输入重算、保存后的原有数值/公式/样式/数据验证/冻结窗格，以及全部新增区域的渲染。未运行桌面 Excel 或 NPU 性能测量。原始二进制保存在 [变更前归档](../examples/archive/pre-pd-contention-20260920/)，历史 manifest 只调整路径，原 SHA256 不变。
