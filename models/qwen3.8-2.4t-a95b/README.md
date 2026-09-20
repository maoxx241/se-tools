# qwen3.8-2.4t-a95b

[模型配置](config.json)。配置来源和 revision 见[证据记录](../../analysis/EVIDENCE.md)，文件哈希见[快照清单](../../analysis/snapshot-manifest.json)。

[权重与通信工作簿](qwen3.8-2.4t-a95b-analysis.xlsx)，2026-09-04 历史快照；包含 Prefill / Decode 独立输入与公式。默认场景与后期 256K 表不同，见[分析记录](../../analysis/SESSION.md)。

通信量按[HCCL 校正说明](../../analysis/HCCL.md)解读，历史文件中的通信组总和不代表单链路流量。

## PD 带宽争用耗时（2026-09-20）

Prefill / Decode 原通信区下方已增加耗时评估。黄色 E 列可逐通信事件填写实际有效带宽（GB/s），默认 H 列输入可统一调整；原生公式自动重算。基线 forward 实测耗时未填时只显示毫秒增量。默认 100 GB/s、10% 带宽降幅是示例。

参见[输入位置、源码与公式](../../analysis/PD-CONTENTION.md)。本次保留原有参数、权重、通信量和缓存值；[历史原始文件](../../examples/archive/pre-pd-contention-20260920/)单独留档。

新增「分钟场景」页：本模型 EP32 / EP256 的 24 个 TPOT / 输入长度 / TTFT 组合，按有限 KV 窗口分列部分和全 EP 重叠次数、基线一分钟工作量增加时间。KV 与 EP 带宽在 N5 / N7 独立编辑，初始为 4800 GB/s。批量、P/D 组数、错峰批数和平均产出 token 数也可编辑；相位平均模型和边界见[一分钟场景说明](../../analysis/PD-MINUTE.md)。
