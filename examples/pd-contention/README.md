# PD 带宽争用示例

[公式、源码与使用说明](../../analysis/PD-CONTENTION.md) · [EP32/EP256 Excel](../kv-ep-sweep/kv-ep32-ep256.xlsx) · [逐模型 Excel](../../models/)

100 GB/s 是可替换示例。带宽降低 10% 时，受影响带宽项耗时增加 11.111%；完整 forward 百分比需要无争用实测基线。

逐模型表在原通信表下方新增黄色 **E 列有效带宽 GB/s**；P/D 分开填写，默认 H 列输入可统一控制。EP 对比表的 **规格汇总 M58:N69** 可按模型、EP 和 P/D 单独填带宽，B55/D55 控制默认值。修改后原生 Excel 公式自动更新。

- `results.json`：六模型 × EP32/256，仅 MoE EP 的每 Rank 毫秒估算。
- `model-results.json`：七份模型工作簿共 14 个阶段，使用各表原有参数，含其已列的通信事件。`region` 是新增区域的行位置。
- 两者默认场景不同，不应直接比较全部毫秒总量。V4.1 是 PD 条件评估，KV 拉取仍为此前快照的规划值。

```sh
node scripts/export-analysis.mjs contention
python3 scripts/verify-contention.py
```
