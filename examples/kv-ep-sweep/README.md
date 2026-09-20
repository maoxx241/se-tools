# EP32 / EP256：通信量、DP 域和 KV 存取

[下载 Excel](kv-ep32-ep256.xlsx) · [计算公式与源码](../../analysis/KV-EP-SPECS.md) · [完整数值](results.json)

2026-09-20 增加 [PD 带宽争用耗时](../../analysis/PD-CONTENTION.md)：规格汇总第 51 行起，黄色 **M58:N69** 可分别填写每个模型、每档 EP 的 P/D 实际有效带宽（GB/s），B55/D55 是默认带宽。10% 降幅、覆盖率、暴露系数及实测 forward 基线也可改，相关耗时自动重算。100 GB/s 仅为示例。

下文 KV 和字节公式是 2026-09-12 固定版本快照；V4.1 新 main 已有 PD 相关改动，本次没有验证端到端迁移，因此原拉取仍保留规划值，旧版本“不支持”说明不代表对新 main 的结论。

主表以通信量为核心。P 每 DP 16,384 tokens/Step，D 每 DP 128 tokens/Step；EP 增大时 DP 域一起扩大。P 按 AllToAllV 加显式 Ring 计数交换，D 按 A3 BF16 MC2 FullMesh 已建模远端写入量。均衡路由，主干网络；不是物理链路实测。

## EP 通信

单位 GiB/Step/EP 组。TP 固定：P 全部 TP8；D 的 DS/GLM TP1、Kimi/Qwen TP8。

|模型|EP|P DP|D DP|P 全组 tokens|D 全组 tokens|P AllToAllV 合计|D MC2 已建模量|
|---|---:|---:|---:|---:|---:|---:|---:|
|DS V4 10T|32|4|32|65536|4096|620.571|40.230|
|DS V4 10T|256|32|256|524288|32768|5111.580|330.985|
|DS V4 Pro|32|4|32|65536|4096|620.571|40.230|
|DS V4 Pro|256|32|256|524288|32768|5111.580|330.985|
|GLM5.3|32|4|32|65536|4096|871.946|56.851|
|GLM5.3|256|32|256|524288|32768|7176.544|467.649|
|KimiK3|32|4|4|65536|512|2495.805|20.290|
|KimiK3|256|32|32|524288|4096|20550.410|166.994|
|Qwen3.8 2.4T|32|4|4|65536|512|1782.674|14.636|
|Qwen3.8 2.4T|256|32|32|524288|4096|14673.955|120.390|
|DS V4.1 Flash|32|4|32|65536|4096|290.682|19.115|
|DS V4.1 Flash|256|32|256|524288|32768|2395.605|157.274|

EP256 的 DS V4 / V4.1、Kimi 为条件规格：原始 384 / 896 专家不整除 256，估算使用 512 / 1024 物理专家槽的均衡映射，均需 128 冗余槽和 EPLB/路由实现验证。其余行也未做实机兼容性测试。V4.1 的 EP 量计 40 层 MoE、TopK=6、H=5120；Engram 的独立通信未并入此表。

每 DP 固定工作量时，全组 token 数增加 8 倍，纯 routed payload 增加 255/31≈8.226 倍；每 Rank 平均 payload 增加约 2.82%。计数交换另含通信域规模项。EP 已包含 DP，结果不能再乘一次 DP。若固定全组负载，应将 EP256 的每 DP token 输入除以 8 后再比较。

## 配套 KV

256K 输入，16 请求/P DP。单位 GiB/批，全部 P/D TP Rank 合计。两档 EP 的单请求/单 P DP 批 KV 相同；Excel 另列 P 整组的 64 / 512 请求总量。存是生成的 KV 行加一份最终辅助状态；保留是阶段末有效数据。前五模型的取是 Mooncake 一次 P→D 交接的张量估算，含块取整及 conv 扩展槽；**V4.1 的取仅为规划值，指定 main 尚未支持 KV 迁移**。

|模型|P 生成（存）|P 阶段末保留|P→D 拉取（取）|
|---|---:|---:|---:|
|DS V4 10T|2232.434|281.394|35.392|
|DS V4 Pro|2232.434|281.394|35.392|
|GLM5.3|2976.000|2976.000|372.000|
|KimiK3|870.693|870.693|871.227|
|Qwen3.8 2.4T|744.749|744.749|745.046|
|DS V4.1 Flash|1370.203|90.828|11.432（规划）|

DS 的生成量包括整个 prompt 的 SWA 行；常驻/拉取仅保留尾部且 D 只取一份 TP 副本。Kimi/Qwen 的拉取略大于生成，来自整块传输和 conv 额外 7 槽。SWA 传输是尾块上界。这里不累计 chunk 状态覆盖写或 Decode attention 循环读取，因此不能把生成量当硬件 HBM 事务或缓存池容量。GLM 新案例统一为 78 主干层/21 Indexer，排除旧表混入的 MTP 层。

V4.1 按 3 个 C2 + 1 个 C1 源缓存、40 层 SWA、3 页 32 行 FP32 ring 单独计算，跨层共享的 KV/Indexer 不重复计数。P 生成与保留含 TP8 副本，D 按 TP1 规划拉取。交接假设为 P 完成 S tokens 后的快照，无前缀命中、不含 draft；C2 的物理页是 B/2 行，只传完成的压缩对，ring 规划按完整 32 行页。其 JSON `pullBytes` 为 `null`、数值另存 `plannedPullBytes`，避免将规划量误用为已有实现流量。

|V4.1 范围|请求数|P 生成 GiB|P→D 规划取 GiB|
|---|---:|---:|---:|
|单请求，全部 TP Rank|1|85.637695|0.714478|
|单 P DP 批|16|1370.203125|11.431641|
|EP32，整个 P 组|64|5480.812500|45.726563|
|EP256，整个 P 组|512|43846.500000|365.812500|

迁移规划不含共享 allocator 页填充、请求/协议元数据、Engram 历史和 draft 状态；这些需适配器实现后再核对实际块列表和传输长度。[逐组件公式与固定源码](../../analysis/KV-EP-SPECS.md#deepseek-v41-flash)记录了这些条件。

## 复算

```sh
node scripts/export-analysis.mjs kv-ep
npm test
python3 scripts/verify-kv-ep.py
# 可选 authoring runtime，要求 @oai/artifact-tool
node scripts/build-kv-ep-workbook.mjs outputs/kv-ep
```

Excel 的黄色输入直接驱动公式：prompt、block size、speculative conv 槽、批请求数、P/D Step tokens、通信 dtype 和 scale。默认 BF16；INT8 dispatch 要同时设置 1 B 数据和 4 B scale，combine 保持 BF16。MC2 ordinary/hierarchy/A5 需独立分支公式，不能沿用 FullMesh 打包值。原始历史案例保持不变。
