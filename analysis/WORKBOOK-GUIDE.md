# 表格使用指南与计算细节

本页对应仓库中已提交的工作簿，方便未参加原始讨论的读者填写参数、核对数据和复算。文档整理于 2026-09-28；H2D/RH2D 的实现依据仍是 [2026-09-22 固定源码快照](host-transfer-sources.json)，没有把文档日期当成新的源码验证日期。

## 选哪份表

| 需要回答的问题 | 工作簿 / 说明 |
| --- | --- |
| 各模型权重和 TP/EP 切分 | [models/](../models/) 中对应模型的 Prefill / Decode 表 |
| EP32/256 的通信量、Prefill 生成 KV 与 P→D 拉取量 | [kv-ep32-ep256.xlsx](../examples/kv-ep-sweep/kv-ep32-ep256.xlsx)，参见 [KV-EP-SPECS](KV-EP-SPECS.md) |
| KV 传输与 forward 通信争用，对 TPOT 和 Decode 吞吐的影响 | 上述总表及逐模型表的 `分钟场景`，参见 [PD-MINUTE](PD-MINUTE.md) |
| V4.1 Engram offloading 的 P/D H2D | [engram-h2d-prefix-rh2d.xlsx](../examples/host-transfers/engram-h2d-prefix-rh2d.xlsx) 的 `Engram H2D` |
| 前缀存在 D host pool，从 D 拉到 P 的 RH2D | 同一工作簿的 `Prefix RH2D`，参见 [HOST-TRANSFERS](HOST-TRANSFERS.md) |

不同表统计的方向和时间范围不同。`forward` 是一次调度 step，`batch` 是每 P DP 一批完整请求，`EP` 是当前规格的一整套 EP 域。例如 Engram 完整 prefill 批次的字节数，不能与一次 decode forward 的字节数直接比较。

## H2D/RH2D 两页表：参数填写位置

以下位置对应已提交的两页表。百分比在 Excel 中填 `50%`，CLI JSON 中填 `0.5`。

| Sheet | 单元格 | 输入及作用 |
| --- | --- | --- |
| Prefix RH2D | D5 | `Requests / P DP`，默认 16；也联动 Engram 的完整 prefill 批次 |
| Prefix RH2D | H5 | `Prefix hit %`，默认 100%；决定命中多少前缀 token，随后按 pool alignment 向下对齐 |
| Prefix RH2D | L5 | `P-local reuse % requests`，默认 0%；整请求已在 P 本地就绪、无需远端拉取的比例 |
| Prefix RH2D | D6 | `Base block tokens`，默认 128；联动各组件 raw-token block |
| Prefix RH2D | H6 | `Extra conv slots`，默认 0；仅改变卷积状态 buffer 大小，不会自动启用投机解码或 draft |
| Prefix RH2D | B54:D59 | 每模型 P TP、D TP、pool alignment；依次是 DS V4 10T、Pro、GLM、Kimi、Qwen、V4.1 |
| Prefix RH2D | B63:B65 | 三种输入长度，默认 16,384 / 262,144 / 1,048,576 tokens |
| Prefix RH2D | D63:D64 | 两种 EP，默认 32 / 256；两页共用 |
| Prefix RH2D | C72:M98 | 缓存组件层数、维度、字节宽度、压缩比例、块、窗口、TP 切分和 state 行数 |
| Engram H2D | H5 | P 每 DP 每步 token 数，默认 16,384；完整 prefill 的 metadata 轮数也使用它 |
| Engram H2D | L5 | D 每 DP 每步 token 数，默认 128；不是本页的 P 请求数 |
| Engram H2D | D6 / H6 | Engram 的 P TP / D TP，默认 8 / 1 |
| Engram H2D | L6 | 每节点 ranks，默认 8；TP 组必须完整地位于节点内 |
| Engram H2D | D7 | CPU-served vector fraction，默认 100%；只缩放向量搬运，不同比缩放 ID 和 metadata |
| Engram H2D | H7 | H2D bytes/element，当前 BF16 路径为 2；CPU 表 FP8 不代表这里应填 1 |
| Engram H2D | D8 | device metadata，1 为 HCCL 元数据路径，0 为 CPU metadata |
| Engram H2D | B36:E36 | V4.1 Engram 表数、最大 n-gram、每 n-gram heads、向量维度 |

`Engram H2D!D5` 是对第二页请求数的引用。该页前五个模型的参数行用于说明归档 config 没有 Engram；主结果区固定使用第 36 行的 V4.1 参数。

两页的 TP 输入分别服务于 offloading 与 prefix pooling。若要比较同一套新的 V4.1 部署，应同时修改 Engram 的 D6/H6 与 Prefix 的 B59/C59。CLI 的 `hostTransferReport` 中，`prefillTP/decodeTP` 覆盖项用于 Engram；prefix sweep 保持各模型既定 TP。需要代码中改变 prefix TP 时直接使用 `prefixTransfer`，见下文。

模型表 F54:F59 的 `OK` 仅表示该处已实现的整除、块对齐和 Kimi/Qwen P/D TP 检查通过。它不代表容量、设备支持或完整启动校验通过。修改 pool alignment 时使用运行时参与缓存组的 LCM，单位是 **raw tokens**，不是压缩缓存的物理行数。

## 一步步核算 V4.1 的 16K 前缀

基线为 100% 命中、P TP8、每 P DP 16 请求、无 P 本地复用。一个请求在一个 P rank 上需要：

| 缓存组件 | 算式 | Bytes / P rank / request |
| --- | --- | ---: |
| SWA | 40 层 × 128 行 × 512 元素 × 2 B | 5,242,880 |
| C2 shared KV | 3 个源 × 8,192 行 × 512 × 2 B | 25,165,824 |
| C2 shared Indexer | 3 个源 × 8,192 行 × (128 + 2) B | 3,194,880 |
| C1 shared KV | 1 个源 × 16,384 行 × 512 × 2 B | 16,777,216 |
| C1 shared Indexer | 1 个源 × 16,384 行 × (128 + 2) B | 2,129,920 |
| 合计 | 上述五项相加 | **52,510,720** |

因此：

```text
单请求 / rank = 52,510,720 B = 50.078125 MiB
单请求 / 全部 P TP = 50.078125 × 8 = 400.625 MiB
每 P DP 的 16 请求批次 = 400.625 × 16 / 1024 = 6.259765625 GiB
EP32 的 P DP 数 = 32 / 8 = 4
EP32 整批 = 6.259765625 × 4 = 25.0390625 GiB
EP256 整批 = 6.259765625 × 32 = 200.3125 GiB
```

这五项只包含 cacheable prefix planes。私有 compressor ring 和 Engram history 的恢复不包含在 52,510,720 B 内；完整恢复所需的额外处理见 [实现边界](HOST-TRANSFERS.md)。

## 改一个参数，哪些量会变

| 改动，其余保持默认 | 应看到的变化 |
| --- | --- |
| EP32 → EP256 | 每 DP 工作量不变时，P DP 数及整 EP 字节数都乘 8；单 rank、单 DP 的字节数不变 |
| P batch 请求数 16 → 8 | RH2D 批次量和 Engram 完整 P 批次的向量量减半；P/D 单次 forward 仍由各自 step tokens 决定 |
| P-local reuse 0% → 50% | RH2D 批次量减半；每个远端命中请求自身的布局和字节数不变 |
| Prefix hit 100% → 50% | 历史缓存随对齐后的命中长度变化；已达到窗口大小的尾部缓存、单个递归端点状态通常不变，因此总量不一定减半 |
| Prefix hit → 0% 或 P-local reuse → 100% | RH2D 批次量归零，原因分别是无远端前缀可取、或请求已在 P 本地就绪 |
| Engram CPU-served vectors 100% → 50% | 向量 H2D 减半；IDs/order/metadata 保持原计算。该输入模拟额外缓存或混合驻留，不声称当前源码已有此机制 |
| Qwen 的 P/D TP8 → TP4 | 4 KV heads 从每 head 两份副本变成一份；全部 TP 合计 GQA 缓存减半，全部 TP 合计递归状态不变 |

以 256K、16 请求/P DP 为例，下面是一个 P rank 的 RH2D 批次量。它不是整个 EP 的总量。

| 模型 | 100% prefix hit，GiB | 50% prefix hit，GiB |
| --- | ---: | ---: |
| DS V4 10T | 35.175293 | 17.786621 |
| DS V4 Pro | 35.175293 | 17.786621 |
| GLM5.3 | 372.000000 | 186.000000 |
| Kimi K3 | 108.837021 | 54.837021 |
| Qwen3.8 2.4T | 93.093918 | 47.093918 |
| V4.1 Flash | 11.347656 | 5.712891 |

GLM 在这个布局下全是按历史长度增长的组件，命中减半时总量也减半。其余模型的尾部或端点状态构成固定项。它们不能通过“上一格总数 × 命中率”正确复算。

## 从字节换算时间，及与吞吐表的关系

按相同资源范围统计的有效带宽 B，以十进制 GB/s 填写，则传输带宽时间为：

```text
transfer ms = bytes / (B × 10^6)
```

单 rank 的 bytes 应配单 rank 的有效带宽；节点聚合 bytes 应配节点共享资源的聚合带宽。整 EP 字节数除以单卡带宽会把并行搬运误算成串行。H2D 使用 host/device 路径的有效带宽，RH2D 使用远端 host 到本地 device 路径的有效带宽，不能直接拿 MoE EP 通信带宽代替。

两页 host-transfer 表目前输出字节预算，没有自动增加到旧表的 forward 耗时。Engram 每步 lookup 与旧 PD 交接的触发节奏不同；池化命中又会改变 P 的实际计算量和交接节奏。若要耦合，需要重新给定持续时间、请求节奏、共享资源和暴露到关键路径的比例。

旧 `分钟场景` 的关键计算关系是：

```text
受影响通信带宽降 10%：该通信时段变成原来的 1/0.9，增加 11.111…%
仅在 KV 时间窗口与通信段重叠时计入增时
Total overhead = EP overhead + TP overhead
Decode throughput loss = Total overhead / (60000 + Total overhead)
```

这里 `Total overhead` 的单位为 ms，表示基线一分钟工作量多花多少时间。固定 batch 和平均每步产出时，TPOT 从 10 ms 变成 11 ms，对应吞吐降幅为 9.0909%，不是 10%。服务端整体 QPS 还取决于 P 侧供给和排队；详细的窗口与部分层重叠公式见 [PD-MINUTE](PD-MINUTE.md)。

## 在仓库内复算

```bash
# 默认 10 个 Engram + 36 个 prefix 案例
node scripts/export-analysis.mjs host-transfers > /tmp/host-transfers-default.json

# 只把 prefix hit 改成 50%，其余默认值保持不变
node scripts/export-analysis.mjs host-transfers \
  examples/host-transfers/half-prefix.json > /tmp/host-transfers-half-prefix.json

# 检查仓库中的默认发布件，而非任意手工修改后的表
python3 scripts/verify-host-transfers.py examples/host-transfers/engram-h2d-prefix-rh2d.xlsx
```

在 Node 模块中改变单模型拓扑的例子：

```js
import { loadPrefixModels, prefixTransfer } from './lib/host-transfers.mjs';

const qwen = (await loadPrefixModels()).find(m => m.profile === 'qwen38');
const result = prefixTransfer(qwen, {
  inputTokens: 262144, ep: 32,
  prefillTP: 4, decodeTP: 4,
  prefixHitFraction: 0.5, requestsPerDP: 16,
});
console.log(result.dpBatchBytes / 2 ** 30); // GiB / P DP batch
```

Excel 中的手动改动不会自动写回 CLI JSON，CLI 参数也不会自动改写已保存的 Excel。发布其他场景时应把使用的参数一并保存。`verify-host-transfers.py` 用相邻 `results.json` 对照默认发布件；手动修改输入后与默认 JSON 不同是预期现象。

实现索引：[`lib/host-transfers.mjs`](../lib/host-transfers.mjs) 管数值，工作簿 builder 管原生公式与排版，[tests](../tests/host-transfers.test.mjs) 覆盖数据方向、TP 副本、EP 扩展与状态边界，[source manifest](host-transfer-sources.json) 记录固定源码版本。`outputs/` 是本地构建目录，供其他读者下载的正式文件位于 `examples/` 和 `models/`。
