# H2D / RH2D 数据量

[下载 Excel](engram-h2d-prefix-rh2d.xlsx)，共两页：

- **Engram H2D**：V4.1 token lookup；P/D 单次 forward 与完整 P 批次，单卡、节点、EP 总量。
- **Prefix RH2D**：六模型 × EP32/EP256 × 16K/256K/1M，D host pool → P device，默认前缀命中 100%。

黄色输入和全部计算区域均可编辑；保留可见网格，未隐藏行列或设置工作表保护。每 P DP 默认 16 请求；支持调整 TP、命中比例、P 本地复用、块对齐、向量与缓存维度。

详细计算和实现边界见 [HOST-TRANSFERS.md](../../analysis/HOST-TRANSFERS.md)。默认字节结果见 [results.json](results.json)；可运行 `node scripts/export-analysis.mjs host-transfers` 复算。
