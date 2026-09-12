# 用 HCCL 校正通信量口径

核对版本：[cann/hccl](https://gitcode.com/cann/hccl) 提交 `170ddeec539b4d693028ce6e0cf5c58933e4d46d`，2026-09-12 获取。这是本次分析依据，不表示历史服务使用了该版本。文件哈希见 [hccl-sources.json](hccl-sources.json)。

## API count 的基数

令 `p` 为同一通信域的 Rank 数、`c` 为 API 的 count、`s` 为通信 dtype 字节宽度：

| API | `c` 的含义 | 本 Rank 输入字节 | 本 Rank 输出字节 |
| --- | --- | ---: | ---: |
| AllReduce | 输入/输出完整 Tensor 元素数 | `c·s` | `c·s` |
| AllGather | 本 Rank 输入 `sendCount` | `c·s` | `p·c·s` |
| ReduceScatter | 本 Rank 输出 `recvCount` | `p·c·s` | `c·s` |
| AlltoAll | 发给**每个 Rank**的 `sendCount` | `p·c·s` | `p·c·s` |
| AlltoAllV | 各目的 Rank 的 `sendCounts[j]` | 各项之和乘 `s`；地址跨度另看 displacements | 根据本 Rank 的 `recvCounts` |

依据：[公开头文件](https://gitcode.com/cann/hccl/blob/170ddeec539b4d693028ce6e0cf5c58933e4d46d/include/hccl.h#L22)、[AlltoAll 参数及示例](https://gitcode.com/cann/hccl/blob/170ddeec539b4d693028ce6e0cf5c58933e4d46d/docs/zh/api_ref/comm_op_interface/HcclAlltoAll.md)。

原表 `B` 是本 Rank Collective **输入**数据量。不能把 profiling 的 `recvCount` 或 per-peer `sendCount` 直接填入：ReduceScatter / AlltoAll 需要先乘 `p`。AlltoAllV displacements 是地址偏移，不是额外发送的 padding；实际 count 包含 padding 时才计入 payload。

## Ring、端点数据量与物理链路

以下 `B` 一律表示本 Rank 输入字节，AllGather 是本地 shard。发送只计一侧，`k` 为该事件每 Step 调用次数。

| 操作与假设 | 本 Rank 发送字节/次 | 全组发送字节/次 |
| --- | ---: | ---: |
| 理想等分 Ring AllReduce | `2(p−1)B/p` | `2(p−1)B` |
| 理想 Ring AllGather | `(p−1)B` | `p(p−1)B` |
| 理想等分 Ring ReduceScatter | `(p−1)B/p` | `(p−1)B` |
| 等分 AlltoAll，跨 Rank 端点数据 | `(p−1)B/p` | `(p−1)B` |

全组每 Step = 全组每次 × `k`。Ring AllReduce 分为 ReduceScatter / AllGather 两阶段。HCCL [Ring 说明](https://gitcode.com/cann/hccl/blob/170ddeec539b4d693028ce6e0cf5c58933e4d46d/docs/zh/user_guide/coll_algo_intro/Ring.md)中的 AllGather `n` 是汇聚后的总数据量，与 API 本地 `sendCount` 不同；以上是基于其算法过程的推导。

例如 TP8，每 Rank AllReduce 输入 100 MiB，每 Rank 理想发送 175 MiB，全组 1400 MiB。不能把接收再加一次得出 2800 MiB。

原表这四个系数在输入口径正确时成立，不需要统一换成另一个“昇腾系数”。需要改正的是将它们当成所有 HCCL 算法、所有物理链路的精确测量。公共 API 明确要求 `algorithm=ring` 或 AlltoAll 的 `direct` 端点模型，未知算法报错；`physicalLinkBytes` 保持 `null`。

Ring 不等长块的逐 Rank 负载未必相等。公共 API 以该理想过程的组级总量为基准，将单 Rank 结果标为平均值分数，不用整数除法截断字节数。

## AlltoAllV 使用发送矩阵，排除自拷贝

令 `C[i][j]` 为 Rank i 上发给 Rank j 的元素数：

```text
Rank i 跨 Rank 发送 = sum(C[i][j] * s, j != i)
Rank j 跨 Rank 接收 = sum(C[i][j] * s, i != j)
全组端点发送        = sum(C[i][j] * s, i != j)
Rank i 自拷贝       = C[i][i] * s
```

`sendCounts_i[j]` 应与 `recvCounts_j[i]` 配对。示例使用相同 send/recv dtype，公共函数检查可选接收矩阵。非均衡 Rank 的发送量不能简单乘 `p`。

依据：[AlltoAllV API](https://gitcode.com/cann/hccl/blob/170ddeec539b4d693028ce6e0cf5c58933e4d46d/docs/zh/api_ref/comm_op_interface/HcclAlltoAllV.md)；[AICPU AlltoAllV 实现](https://gitcode.com/cann/hccl/blob/170ddeec539b4d693028ce6e0cf5c58933e4d46d/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_ubx_all_to_all_v_mesh_1D.cc#L291)对自己执行 LocalCopy，对其他 Rank 执行通信；[CCU AlltoAll](https://gitcode.com/cann/hccl/blob/170ddeec539b4d693028ce6e0cf5c58933e4d46d/src/ops/all_to_all_v/algorithm/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d.cc#L137)同样区分本地拷贝与远端 Write。

[示例矩阵](../examples/hccl/alltoallv.json)为 `[[100,10,0],[30,200,5],[2,0,300]]`，元素 2 B。各 Rank 跨 Rank 发送为 `[20,70,4] B`，全组 94 B；对角线 1200 B 自拷贝单列。取任意一个 Rank 乘 3 都会出错。

EP32 的 `31/32` 仅在均匀路由假设下成立。如果 EP 行已经乘了跨 Rank 比例，后续不能再乘一次 `(EP−1)/EP`。dispatch / combine 需分别记录；两方向精度或元数据不同，不能直接乘 2。历史案例保留 hidden-only、两方向等宽的假设。

## 自适应算法与多层拓扑

[HCCL_ALGO](https://gitcode.com/cann/hccl/blob/170ddeec539b4d693028ce6e0cf5c58933e4d46d/docs/zh/user_guide/hccl_env/HCCL_ALGO.md)说明算法依赖产品形态、数据量和 Server 数，环境变量按层配置；部分小数据场景仍会自适应选择。`level1:ring` 不能理解为整个通信域只有一个 Ring。

同一提交的 [AllReduce selector](https://gitcode.com/cann/hccl/blob/170ddeec539b4d693028ce6e0cf5c58933e4d46d/src/ops/all_reduce/selector/all_reduce_auto_selector.cc#L430)根据 dtype、拓扑层数、对称性和数据量选择 SoleNHR、MeshNHR、流水等实现。这是不能将 Ring 当默认事实的实现依据。本次没有为这些路径臆造通用倍率。

端点数据量不等于 RoCE 总量，也不等于 HCCS/UB/HBM 总量。拆分这些量需要每层算法、Rank→设备/Server 映射、链路类型、分片、转发/远端读、并发 Plane 和协议开销。EP32 组可能跨越多个 TP8 组，两种组的总量不能直接相加称为单链路带宽需求。

FP4 权重的 0.5 B/元素不意味着 HCCL 有 0.5 B 的通用通信 dtype。通信应使用实际 API dtype 与打包后的 count；权重、激活、索引、scale 分别取宽度。

## 用 Profiling 校准

按模型 Step、通信域 ID、Collective 和调用次序对齐，核对每个 Rank 的 API count/dtype；AlltoAllV 保留各 Rank send/recv 数组。用实测事件替换清单中对应项，避免同时计入估计与实测。

HCCL [Profiling 说明](https://gitcode.com/cann/hccl/blob/170ddeec539b4d693028ce6e0cf5c58933e4d46d/docs/zh/user_guide/perf_analysis/profiling_op_behavior.md)区分 Host 下发、Device Collective、Notify 和数据任务。它指出 RDMASend 可能是同步任务，所述同步任务为固定 4 B；数据 RDMASend 的 duration 可能只是 WQE 下发耗时。因此：

1. 根据任务语义、src/dst、size 及相邻同步关系区分数据、控制和自拷贝。不要按名字累计所有 RDMASend，也不要把所有 4 B 任务直接丢弃。
2. 在发送端计一次，避免源/目的 Rank 重复统计；多 Plane 按任务标识和通信域对齐，不能只按时间戳去重。
3. 用 Device Collective 时间区间评估事件耗时，分析等待和重叠；不将所有 task duration 相加当 wall time。
4. 链路带宽按对应链路/方向字节及时间窗口计算，并声明 payload 或物理计数器口径。全组字节除以单卡标称带宽不是精确耗时。

当前没有假定某版 msprof CSV schema，也没有自动物理流量解析器。应保留版本、硬件、通信域和 Rank 映射、算法/环境变量、Step ID、count/dtype、src/dst、Plane、任务语义及 Device 时间。

## 本次校验范围

`npm test` 验证 count 基数、单 Rank、空数据、非均衡 AlltoAllV、自拷贝、接收配对、大整数和未知算法拒绝。`verify-archive.py` 比较公共计算与归档表的 84 个通信字节值，并核对哈希及空白通信能力区。

历史 Excel 保持原样，仍有旧的“有效负载/通信组发送”术语和不同组相加的合计，应按本文解读。新构建脚本用“建模发送”标签，取消不同组的总和；单 Rank 事件和仅在相同参与 Rank/调用口径下使用。本次未执行 HCCL 编译、NPU 集群或链路性能测试。
