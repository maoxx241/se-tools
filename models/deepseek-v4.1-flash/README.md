# DeepSeek V4.1 Flash

2026-09-12 新增的模型分析。配置来自 [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277)，实现分析固定到用户指定 VA 仓库 main 的 [`1933f86`](https://github.com/GDzhu01/vllm-ascend-v41-private/tree/1933f86cbed1ee69fc1e1a9e0b99ef1c2ff1195c)。采集时两仓库均为公开可读。

- [可编辑 Excel](deepseek-v4.1-flash-analysis.xlsx)：Prefill / Decode 各自的权重、参数和通信事件。
- [config.json](config.json)：官方原始配置。
- [权重元数据](weight-metadata.json)：48 个分片的 header 哈希、96,085 个 Tensor 的 116 类形状/精度/数量、完整覆盖及大小校验；仅通过 HTTP Range 读取头部，没有下载权重载荷。
- [通信复算结果](communication-example.json)、[实现分析与公式](../../analysis/DEEPSEEK-V4.1.md)、[来源及哈希](sources.json)。
- [EP32 / EP256 通信与 KV 存取扩展](../../examples/kv-ep-sweep/)：逐组件生成、保留、P→D 规划量和 DP 域汇总；Excel 公式可编辑。当前 main 未支持迁移，规划量与实际可用路径分开标注。

模型为 40 层、H=5120、64 个 Attention Head、Head Dim=512；每层 384 个 Routed Expert、TopK=6、1 个 Shared Expert。包含两张 Engram 表、32 层视觉编码器、3 层 DSpark draft（128 Expert、TopK=3）。参数名称、shape、norm、scale 分别建模。

## 权重结果

下表按官方文件的存储 dtype 计算，包含 scale，单位为字节和二进制 GiB。FP4 的逻辑 shape 展开为每元素 0.5 字节；其 Safetensors header 用 I8 容器存放两个 E2M1 数值。

| 部分 | 字节 | GiB |
| --- | ---: | ---: |
| 主体文本（不含 Engram） | 298,309,535,168 | 277.822404 |
| Engram 表及 Gate | 203,073,076,240 | 189.126540 |
| DSpark | 7,932,874,632 | 7.388065 |
| 视觉、Aligner、图像标记 | 970,536,960 | 0.903883 |
| 合计 | 510,286,023,000 | 475.240893 |

总字节精确匹配官方 index 的 `metadata.total_size`，全部分组的存储 shape/dtype/count 也逐项匹配。这里的“文本主体”是文件分类，不等价于模型卡的参数量分类。

Excel 使用官方源格式计算权重字节和分片份额，不能当成 VA 实际 HBM 驻留量。尤其是 Engram，VA 可选 BF16 / INT8 在设备，FP8 / MXFP8 在 CPU；Gate 使用 BF16，C2 投影使用 F32，DSpark main projection 使用 BF16。当前 VA loader 还使用 `quant_model_weights.safetensors.index.json` 并校验 Engram Gate BF16；提供的官方原始文件名和部分 dtype 与此不同，**本分析不表示官方 checkpoint 可直接在该 main 上启动**。

## 使用

```bash
node scripts/export-analysis.mjs weights deepseek-v4.1-flash
node scripts/export-analysis.mjs communication deepseek-v4.1-flash
MODEL=deepseek-v4.1-flash node scripts/build-analysis-workbook.mjs outputs/models
MODEL=deepseek-v4.1-flash node scripts/verify-analysis-workbook.mjs outputs/models
```

前两条不需要第三方依赖或 NPU；后两条需要 Artifact Tool，见[复算指南](../../analysis/REPRODUCING.md)。权重精度在 Excel E 列修改；激活、Dispatch、Combine 宽度分别修改。节点内分片栏为平均份额，实际连续行分片存在尾部取整。

新增案例使用 Prefill TP8×DP4 / EP32、DSA CP+SP；Decode TP1×DP32 / EP32，SP 和 DSA CP 关闭。每节点 8 Rank，投机步数默认 0。它是可复算的对比场景，不是部署性能推荐。源码支持边界及缺失通信项均写入表内说明。原有 2026-09-04 六模型案例保持原值。

## PD 带宽争用耗时（2026-09-20）

Prefill / Decode 原通信区下方已增加耗时评估。黄色 E 列可逐通信事件填写实际有效带宽（GB/s），默认 H 列输入可统一调整；原生公式自动重算。基线 forward 实测耗时未填时只显示毫秒增量。默认 100 GB/s、10% 带宽降幅是示例。

参见[输入位置、源码与公式](../../analysis/PD-CONTENTION.md)。本次保留原有参数、权重、通信量和缓存值；[历史原始文件](../../examples/archive/pre-pd-contention-20260920/)单独留档。
