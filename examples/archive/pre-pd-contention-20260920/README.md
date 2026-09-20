# 加入 PD 耗时评估前的 Excel

这些文件是 2026-09-20 扩展前的原始字节副本，用于复现历史结果和检查新增区域没有改变原数据。当前可编辑版本仍在 `models/` 与 `examples/kv-ep-sweep/`。

六份早期模型表来自 2026-09-04，V4.1 与 EP 规格表来自 2026-09-12 扩展。`analysis/snapshot-manifest.json` 和 V4.1 `sources.json` 中相应历史 XLSX 记录仅迁移路径，保留原 SHA256，并增加 `original_path`。

`python3 scripts/verify-contention.py` 对比原有数值、公式、样式、数据验证及冻结窗格。此目录不是更新后的带宽输入模板。
