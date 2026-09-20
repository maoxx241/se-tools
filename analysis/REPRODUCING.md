# 复算与扩展

## 使用已有结果

无需生成器，直接打开 `models/<slug>/<slug>-analysis.xlsx`。Prefill / Decode 右侧参数相互独立；修改某项权重的 E 列字节宽度，只影响该项。config 改动不会自动流入已经打开的 Excel，需要重新生成。

逐模型表默认值与最终 256K 表不同，见 [SESSION.md](SESSION.md)。修改通信精度应以实际发送 Tensor 为准；将权重 FP4 改成 BF16 不应顺带修改索引和归约 LSE 的宽度。

PD 耗时区域的黄色 E 列是逐事件有效带宽输入，单位 GB/s；EP 对比表在「规格汇总」M58:N69 可按模型/EP/P/D 分别填值。[PD-CONTENTION.md](PD-CONTENTION.md)列出所有输入位置、公式、来源和边界。无需脚本，直接改 Excel 即可复算；实测 forward 基线未填时只展示毫秒增量。

## 在普通开发环境复算

Node.js 20+，Python 3 用于读 ZIP/XML 的独立归档检查，不需要 npm install：

```bash
npm test
node scripts/export-analysis.mjs weights > /tmp/weights.json
node scripts/export-analysis.mjs communication > /tmp/communication.json
python3 scripts/verify-archive.py
node scripts/export-analysis.mjs contention
python3 scripts/verify-contention.py
```

公共模块可直接 import：

可选的 [GitHub Actions 模板](../examples/github-actions/)可用于持续验证；仓库当前未启用远端 CI。

```js
import { estimateCollective, estimateAllToAllV } from './lib/collectives.mjs';

// HCCL recvCount=128：每 Rank 输出 128 个元素，输入为它的 8 倍。
estimateCollective({
  collective: 'ReduceScatter', algorithm: 'ring',
  count: 128, dtypeBytes: 2, ranks: 8, callsPerStep: 3,
});

// sendCounts 按 [源 Rank][目的 Rank] 排列，单位为元素。
estimateAllToAllV({
  sendCounts: [[100, 10, 0], [30, 200, 5], [2, 0, 300]], dtypeBytes: 2,
});
```

库结果使用 BigInt，使用导出的 `jsonBytes` 序列化。超过 JS 安全整数范围的输入应传 BigInt 或十进制字符串。AlltoAllV 不猜测未知 sendCounts；Ring 之外的算法也不自动回退。

## 生成 Excel

构建及 Artifact Tool 校验是可选路径，需要环境已有 `@oai/artifact-tool`。将 `ARTIFACT_TOOL_ENTRY` 指向该环境的模块入口，或让 Node 能正常 import 该包，再执行 README 中的构建命令。不要把本机依赖目录或个人路径写进仓库。

生成器从仓库模型目录读取 config，不依赖某个会话 ID、Downloads 或临时文件。`MODEL_CONFIG_ROOT` 必须包含 `<slug>/config.json`。默认输出为 `outputs/models/`；若手动指定输出，不要指向归档 `models/`。通信填表脚本接受原有 SOURCE/OUTPUT 参数和可选的 10T、Qwen27 config 覆盖。

在保留当前 Excel 其它内容的前提下重新构建 PD 耗时区域，可运行 `node scripts/update-contention-workbooks.mjs --write outputs/pd-contention`。输出七份逐模型表和一份 EP 对比表，耗时区重置为示例输入；已填真实带宽的文件不要通过此命令更新，应直接在 Excel 复算。`--inspect` 只查看原通信区。历史字节副本位于 `examples/archive/pre-pd-contention-20260920/`。

只按原表风格统一新增区域时，使用 `--style` 代替 `--write`。该模式保留全部数值、公式及已填写的带宽，逐模型表沿用浅蓝表头/细边框/紧凑行高，EP 汇总沿用深蓝白字表头和原黄色输入色。

## 增加模型

V4.1 的完整新增范例见 [DEEPSEEK-V4.1.md](DEEPSEEK-V4.1.md)：`lib/deepseek-v41.mjs` 提供权重 shape、source graph、缓存及通信；`scripts/deepseek-v41-workbook.mjs` 提供对应 Excel 事件。它不修改原有六模型历史案例。新模型应按实际调用路径处理，不能仅改变维度后继承旧模型的所有通信行。

公开 HF 元数据可以独立重取（只下载配置、index 和 Safetensors 头部）：

```bash
python3 scripts/snapshot-hf-metadata.py \
  --repo deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --output outputs/v41-metadata
```

输出目录必须不存在；该脚本联网但不下载 Tensor 数据。离线 `npm test` 用仓库内的分组快照逐项核对 V4.1 shape、dtype、重复数、packed FP4 字节及完整总量；与旧模型“仅 config/code 建模”的证据强度不同。

1. 在 `models/<slug>/` 保存完整配置、来源 URL、revision（缺失则明确注明）和采集时间，登记文件哈希。
2. 在 `lib/model-catalog.mjs` 登记模型。为该模型在 `scripts/model-analysis-specs.mjs` 返回 `{ rows, facts }`；按真实架构拆 Tensor，量化 scale 和 norm 单独列行。
3. 每行需要模块、名称、shape、dtype、字节/元素、重复数和分片规则。不要在维度缺失时随意套用其他 Transformer 的公式。区分源文件 dtype、运行时参数 dtype 和通信 dtype。
4. 在 Excel `communicationEvents` 中记录通信位置、Collective、本 Rank 输入、dtype、组和每 Step 调用次数。新增模型不自动继承 DSA CP、共享专家或 KV 布局。
5. 若纳入 256K 案例，再扩展 `communicationModels`、该模型 facts 和 P→D 函数。27B/10T 目前只有此案例能力，不要宣称有对应权重工作簿。
6. 验证关键 shape/字节数、代表性参数变化、组大小 1、SP 开关、dispatch/combine 精度与计数。最后再做保存、公式和显示检查。

已有 `lib/communication-requirements.mjs` 的固定参数用于历史案例复算；修改它会改变案例，需要更新结果和说明。它不是任意拓扑仿真器。直接复用其 `calculateStepTraffic` 时，适用范围是普通 TP/SP 与当前案例的 DSA 抽象，不含 Fine TP、一般 PCP/DCP 或硬件兼容性验证。

## 接入 HCCL / Profiling

先完成 [HCCL.md](HCCL.md) 的 count 归一化，区分 API Tensor 字节、所选算法 payload、物理链路计数。保存通信域成员、Rank 映射、Step/调用标识、dtype、算法与版本。AlltoAllV 收集全组计数矩阵后再计算不均衡，不能用一个平均路由比代替。

需要链路时延/带宽时，另行采集 Device 时间与链路证据。PD 扩展用可编辑有效带宽做敏感性估算，默认 100 GB/s 不代表标称或实测硬件性能。MFU/MBU 和完整仿真应在计算/访存/通信事件可对齐后扩展。
