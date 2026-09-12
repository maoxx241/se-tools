# 256K P/D 通信需求案例

[历史 Excel](communication-requirements-20260904.xlsx)保存 2026-09-04 最终填写结果。[results.json](results.json)是公共计算模块的可复算数值；所有字节字段为十进制字符串。

条件：输入 262,144 token/请求、16 并发/DP、Prefill 16,384 token/Step、Decode 投机 7 步即 128 token/Step。DeepSeek/GLM Decode 为 TP1+DP32+EP32；Kimi/Qwen 为 TP8+DP4+EP32。仅 Kimi 共享专家切 TP。详情见[场景记录](../../analysis/SESSION.md)。

I 列为固定结果文本，不随修改其他单元格重算。六个已填写模型：DS V4 10T、DS V4 Pro、GLM5.3、KimiK3、Qwen3.8 2.4T、Qwen3.8 27B。MiniMaxH3/WAN2.2 保留模板内容，通信能力区为空。

```bash
node scripts/export-analysis.mjs communication
python3 scripts/verify-archive.py
```

每 Step 数值只针对各自命名的通信组；Attention 与 SP/DSA 行重叠，不可合计。P→D 按一个源 DP 的 16 请求迁移一次，包含案例定义的目的端副本与状态缓冲，不是每 Step 数据量。

这些是模型假设下的 Ring 基准/端点 payload。均匀 EP 路由不等于实际分布，P→D 状态容量也不保证等于 connector 的最小传输量。HCCL 校正见 [HCCL.md](../../analysis/HCCL.md)。归档数值未因术语校正而重写。

## 部分结果

以下是历史假设下的组级数据量，不是链路实测。EP 仅含 routed hidden 的 dispatch+combine，不含共享专家、元数据和 scale。P→D 包含案例定义的目的端副本/状态，不能与 Step 列相加。

| 模型 | EP32 Prefill GiB/Step | EP32 Decode GiB/Step | P→D GiB/批次 |
| --- | ---: | ---: | ---: |
| DS V4 10T | 620.484375 | 38.780273 | 35.175293 |
| DS V4 Pro | 620.484375 | 38.780273 | 35.175293 |
| GLM5.3 | 883.500000 | 55.218750 | 377.500000 |
| KimiK3 | 2495.500000 | 19.496094 | 871.226807 |
| Qwen3.8 2.4T | 1782.500000 | 13.925781 | 745.046143 |
| Qwen3.8 27B | 0.000000 | 0.000000 | 514.396484 |

10T 和 Pro 在本案例中读到的通信相关维度相同，因此上述结果相同；不能据此判断两个模型或权重整体相同。
