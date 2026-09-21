# Engram H2D 与前缀池化 RH2D

配套文件：[两页 Excel](../examples/host-transfers/engram-h2d-prefix-rh2d.xlsx)。黄色单元格为输入，结果是 Excel 原生公式；所有行、列和计算区可见，没有工作表保护。

默认沿用六个模型、P TP8、EP32/EP256，以及每 P DP 16 请求。长度为 16,384 / 262,144 / 1,048,576 tokens，前缀命中比例 100%。EP=TP×DP，因此每 DP 工作量固定时，EP32→256 的整组字节数增加 8 倍，单卡量不变。这是接收端 payload 的数据量，不是实际链路上的协议总量或带宽测量。

## 1. Engram：按 token lookup 搬运向量

使用 V4.1 Flash 的两张 Engram 表，`max_ngram_size=4`、每 n-gram 8 heads、每向量 256 elements：

```text
lookups/token = 2 × (4−1) × 8 = 48
vector bytes/token = 48 × 256 × 2 = 24,576 B = 24 KiB
```

当前 VA main `6bb7aeec` 的 FP8/MXFP8 offload 路径在 CPU 上保存 codes 和 scales，`lookup_local` 在 CPU 解量化并写入 BF16 暂存；随后 `.to(device, dtype=bfloat16)` 才是 H2D。CPU 表的单行 256 codes + 8 E8M0 scales（264 B）不能当作 H2D 的单行字节数（512 B）。BF16/INT8 全表驻设备路径不属于本次 offload 案例。

每个 TP 组只有 leader 提交请求，节点内所有 rank 按 owner 分担返回向量。默认每节点 8 ranks、owner 负载均衡、完整 TP 组在同一节点，所有 DP 持续工作。令 `T` 为每 DP tokens，`Q` 为 lookup/token，`W` 为向量维度，`b` 为 H2D bytes/element，`h` 为 CPU-served vector fraction，`n` 为节点内 ranks，`R` 为调度轮数：

```text
vectors / average rank = T / TP × Q × W × b × h
IDs + order / TP leader = T × Q × (8 + 8)
device metadata / rank = R × Engram tables × (n+1) × 8
total / average rank = vectors + (IDs + order)/TP + metadata
total / node = average rank × n
total / EP = average rank × EP
total / leader = vectors + IDs + order + metadata
```

`metadataOnDevice=1` 对应 HCCL 元数据路径；设 0 对应 CPU metadata。向量双缓冲是空间分配，不会使 H2D 乘 2。源码只按 owner 排序，不对重复 token/hash ID 去重。`CPU-served vectors` 默认 100%；调小它表示额外的 HBM 向量缓存或混合驻留假设，源码当前没有这样的 hot-vector cache，ID 与元数据开销也不会随这个输入一起消失。

默认 P 每步每 DP 16,384 tokens，P TP8：向量 **48 MiB/rank**，加 IDs/metadata 为 **49.500137 MiB/rank**，leader 为 **60.000137 MiB**。D 每步每 DP 128 tokens、D TP1：向量 **3 MiB/rank**，含辅助上传为 **3.093887 MiB/rank**。完整 prefill 批次用 `requests×input length`，metadata 轮数为 `ceil(batch tokens / P chunk tokens)`，不是把完整 prompt 算成一次 forward。

设备之间的查询/返回 AllToAll、TP broadcast，以及反向 D2H 查询 IDs/metadata、CPU 内部解量化访存，均不叠加到 H2D。前五个模型的归档 config 没有 Engram 表，因此在本页标为 0 / no Engram in archived config；`num_hash_layers` 的 MoE 哈希路由不是 Engram lookup。

## 2. Prefix pooling：D host → P device

本页假设命中的缓存已经在 D 节点的 **host DRAM pool** 中，统计 P 侧各目的 rank 从 pool 拉取的数据。不包含先前 D device→host 的写入；如果 pool 实际放在远端 HBM，字节布局可相同，但方向应称 D2D/RD2D。

当前 AscendStore 路径先计算各组缓存的命中范围，再在每个接收 rank 调用 `get` / Mooncake `batch_get_into_multi_buffers`。MLA 的重复 TP rank 可使用相同 pool key，但接收 buffer 仍各自搬运；源码没有在一次 RH2D 后用 TP broadcast 替代这些接收副本。因此本页按 **P TP 的缓存布局** 计算，不能把旧的 P→D、D TP1 的量原样倒过来使用。D 的相同源副本不重复乘一次。

设输入长度 `S`、命中比例 `h`、实际 pool 传输对齐 `A`，默认不做细粒度 partial-chunk 命中：

```text
H = floor(S × h / A) × A
full / compressed rows = H / compression ratio
SWA / compressor tail rows = min(H/B, ceil((window−1)/B)) × B / ratio
recurrent endpoint = 1 state, if H > 0; otherwise 0
component bytes/rank = layers × transferred rows × bytes/row/rank
request bytes/all P TP = sum(component bytes/rank) × P TP
rank batch bytes = request bytes/rank × requests/P DP × (1−P-local request reuse)
P DP batch bytes = rank batch bytes × P TP
P EP batch bytes = rank batch bytes × EP
```

这里的 tail 数来自 prefix coordinator 的 **load mask** 和 `SlidingWindowManager`，只拉命中端点需要的 `ceil((window−1)/block)` 个有效尾块，不沿用旧 PD 交接时 `ceil(window/block)+1` 的保守保留块计数。Full-hit 时 scheduler 会将可直接复用的 token 数减一以重算最后一个 token，但 pool worker 在该分支把传输范围补回完整末块，因此 100% 命中的传输量不是简单的 `S−1` 行。

组件布局：

| 模型 | 可命中缓存 / 状态 | P TP8 的处理 |
| --- | --- | --- |
| DS V4 10T / Pro | SWA、C4/C128 KV、C4 Indexer、压缩器/Indexer FP32 尾部状态 | 每个 P rank 均需一份；两模型当前归档的缓存几何相同 |
| GLM5.3 | 78 MLA、21 Indexer | 每个 P rank 均需一份 |
| Kimi K3 | 24 MLA + 69 KDA conv/recurrent endpoint | MLA 副本 ×8；state 按 TP 切分 |
| Qwen3.8 2.4T | 23 GQA + 69 GDN conv/recurrent endpoint | 4 KV heads 在 TP8 下各复制两份；state 按 TP 切分 |
| V4.1 Flash | 40 SWA + 3 C2 / 1 C1 共享 KV 与 Indexer 源 | 只计 4 个源缓存，不按共享消费者层数重复；各 P rank 需副本 |

**Kimi / Qwen 使用 `mamba_cache_mode=align` 的一个端点状态**，不是每 token 一个 state，也不重复累计 prefill 各 chunk 的中间状态。默认无 speculative/draft；`Extra conv slots` 只改变 state buffer 的大小，不代表打开 Eagle 或改变 prefix 命中规则。默认 P/D 都是 TP8，改动时需匹配状态切分；表中会检查 P/D TP 是否相等。

DSV4 沿用 BF16/A3 cache 布局：物理压缩块 128 行对应 C128 的 16,384 raw tokens，故默认 pool 对齐为 16,384。其他模型给定 128 raw tokens 的整块场景。实际 Mamba state/group 配置可能提升 LCM，模型表中的 `Pool align tokens` 可以直接填写运行时值；必须是各参与组 raw-token block 的公倍数。这里没有把 A5 FP8 cache 的不同布局套入 BF16 案例。

**V4.1 只计算 cacheable prefix planes**。32-row FP32 compressor ring 是请求私有状态，被排除在 prefix-cache hit 外，Engram history 也不是 KV pool plane。因此表中 `State=0` 表示本页没有传这些私有状态，并不表示继续推理不需要这些状态。main 的通用 pool 路径与 V4.1 完整恢复的适配仍需分别确认，不能把本页的字节数解读为端到端迁移已完成。

`P-local reuse % requests` 的单位是整请求比例；设 100% 时 RH2D 批次量归零。默认 16 个请求分别需要远端前缀，未假定这些请求可共享一个本地 destination block；若实际已共享，应使用该输入反映避免的整请求拉取，而不是对 state 乘 token 命中比例。

## 复算与校验

```bash
node scripts/export-analysis.mjs host-transfers
node scripts/export-analysis.mjs host-transfers options.json
node --test tests/host-transfers.test.mjs
ARTIFACT_TOOL_ENTRY=/path/to/artifact_tool.mjs node scripts/build-host-transfer-workbook.mjs outputs/host-transfers
python3 scripts/verify-host-transfers.py examples/host-transfers/engram-h2d-prefix-rh2d.xlsx
```

`options.json` 可覆盖 `lengths`、`eps`、`requestsPerDP`、`prefixHitFraction`、`localRequestFraction`、`prefillChunkTokens`、`decodeTokensPerDP` 等。模型维度来自仓库 configs；Excel 内同样保留了可编辑维度、dtype 字节数、window/block、TP 与 alignment。CLI 字节用 Number 整数表示，当前最大场景仍低于精确整数上限。

构建验证包括：默认 10 个 Engram / 36 个 prefix 场景与独立 JS 公式对照；0/50%/100% 命中、全本地复用、块边界、batch、spec conv slots、block64/128、Kimi/Qwen TP4/8 的联动；导出后重开 XLSX 核对结果并检查两页版式。尚未用真实 NPU profiling 或原生 Excel 应用重算。字节口径与文件完整性验证不等于实测运行支持。

完整源文件路径、固定 revision、SHA-256 和代码定位见 [source manifest](host-transfer-sources.json)。
