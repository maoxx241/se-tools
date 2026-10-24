# 模型显存分析工作簿

这个入口将模型结构、量化格式、部署参数和 Excel 排版分开。输入是仓库已有的公开配置，或一份独立的模型描述；从空白工作簿生成完整结果，不读取旧 Excel，不依赖会话目录或补丁执行顺序。

已有示例是 [Kimi-K3 的公开配置](../models/kimi-k3/config.json)，生成语言模型、含 ViT 两种范围的 Prefill / Decode 共四页。[虚构小规格案例](../examples/model-memory/synthetic.json)用于覆盖混合注意力、Indexer、MXFP、动态量化和不均匀 PP；其中的尺寸和配方不是任何真实模型的规格。

## 直接使用

打开 [Kimi-K3 显存工作簿](../models/kimi-k3/kimi-k3-memory.xlsx)，修改黄色单元格。每页参数独立，权重类型使用下拉框；右侧给出逐结构重复数、单请求 cache、每个 PP stage 和最大单卡合计。

常用输入位于 `R2:R22`：主 TP、独立投影 TP、EP、共享专家 TP、上下文并行、两类 cache dtype、token 数、block size 和当前 PP stage。共享专家 TP 是独立数值默认值，修改主 TP 不会自动覆盖它。其他跟随关系由场景文件的 `links` 声明，生成后仍可在 Excel 单独覆盖。

权重表逐项显示全模型和单卡字节。右侧辅助列分列主权重、scale、附加参数；MXFP4/MXFP8 的 scale 按 TP 后的局部 shape 和 32 元素分组计算。INT4/INT8 使用当前 profile 的每输出通道 16/8 字节附加参数。这些格式是存储计算方案；切换下拉框本身不证明对应算子支持该组合。

`权重 + cache + buffer` 包含表中显式建模的权重、一个请求的缓存及 o_proj gather/transpose buffer。其他激活、图、workspace、分配器和 cache pool 预留不在合计内。

## 复算与重新生成

Node.js 20+ 和 Python 3 即可做数值复算、离线测试和已保存文件检查，不需要下载权重或连接设备：

```bash
npm test
node scripts/export-analysis.mjs memory kimi-k3
node scripts/export-analysis.mjs memory --spec examples/model-memory/synthetic.json
npm run verify:memory
```

重新生成 Excel 需要 `@oai/artifact-tool`。加载器优先使用 `ARTIFACT_TOOL_ENTRY` 或正常模块导入，也可发现 Codex Desktop 的标准 bundled runtime。该运行时不是仓库的 npm 安装依赖。

```bash
npm run build:memory -- --model kimi-k3 --render
node scripts/build-memory-workbook.mjs \
  --spec examples/model-memory/synthetic.json --render
node scripts/verify-memory-workbook.mjs \
  kimi-k3 outputs/memory/kimi-k3/kimi-k3-memory.xlsx
node scripts/verify-memory-workbook.mjs \
  --spec examples/model-memory/synthetic.json \
  outputs/memory/synthetic-hybrid/synthetic-hybrid-memory.xlsx
```

默认构建会遍历输入联动，导出到 `outputs/memory/<id>/`，再读取保存后的 OOXML 核对公式、缓存结果、下拉和冻结窗格。成功后才替换构建目录的目标文件。`--render` 另生成各区域预览；`--no-sweep` 可在已完成联动验收后快速重建。

生成件包括工作簿、确定性的 `memory-results.json` 和 `memory-verification.json`。结果记录配置或模型描述、场景文件、计算代码及工作簿模板的 SHA-256；相同输入可以对比数值结果，不要求 ZIP 文件字节完全相同。默认构建拒绝直接写入 `models/`，正式发布时显式复制已验证的文件。

`verify:memory` 用 Python 标准库检查已保存公式及缓存值，并与现场运行 JS 得到的结果逐项比较。它不执行 Excel 公式。JS 验证器会重新载入文件、恢复经 OOXML 核实的 shared formulas，再执行改参重算；原生 Excel / LibreOffice 的行为应单独验证。

安装 LibreOffice 后，可以在临时副本中修改公开示例的 TP、清空公式缓存并验证真实重算。原文件保持不变：

```bash
python3 scripts/verify-memory-native.py models/kimi-k3/kimi-k3-memory.xlsx
python3 scripts/verify-memory-native.py \
  examples/model-memory/synthetic-hybrid/synthetic-hybrid-memory.xlsx \
  --spec examples/model-memory/synthetic.json
```

## 接入其他模型

完整例子见 [synthetic.json](../examples/model-memory/synthetic.json)。一份描述包含：

- `id`、`schemaVersion: 1`、`evidence`：输出标识、格式版本及真实来源或假设说明。
- `facts`：各类层的编号、head/cache 维度、专家数、卷积长度和上下文上限。层号从 1 开始；注意力和 Dense/MoE 层列表分别必须覆盖全部层。
- `rows`：每项的 `group`、`name`、`shape`、`dtype`、`layerIds` 和 `rule`。首尾模块使用 `placement: first/last`；`instances` 表示该位置的重复数。
- `recipes`、逐行 `recipeDtypes`：明确的配方名称与 dtype 覆盖。未覆盖的行保留声明的基础 dtype。
- `defaults`、`scenarios`：可编辑输入及各场景的 P/D、范围、配方、PP 分层和跟随关系。

分片规则包括 `Replicated`、`EP`、`KDA TP`、`KDA O TP`、`DSA TP`、`DSA O TP`、`LM head TP`、`Embedding TP`、`Other TP`、`Shared TP`。可显式指定 `axis`；默认 O projection 和 Dense/Shared down projection 切第 1 轴，其余切第 0 轴。EP 使用 `CEILING(experts/EP)` 计算最大单卡槽位，保留不能整除和 EP 大于专家数的情况。

结构分组沿用示例中的名称，供公共模板识别；未知分组、缺失层归属、重复 Tensor、非法局部 shape 和未知 dtype 会报错。`partitions` 是各 stage 连续拥有的层数，必须精确覆盖全模型；embedding/ViT 归首 stage，lm_head/末尾 norm 按行声明归末 stage。总占用的最大值取各 stage 完整合计的最大值。

当前 cache profile 覆盖压缩 MLA、可选的按 DCP rank 复制的 Indexer，以及 KDA recurrent/conv state。该 profile 中 Prefill DSA CP 切计算、主 KV 按卡保留；Decode MLA 沿 token 维分摊，Indexer 保持复制。MLA 与 Indexer dtype 独立，C8 的 RoPE 和 scale 单独计入；KDA state 不随历史 token 线性增长。非 DSA 基线仅提供 BF16 cache。

这些布局必须由新模型的源码与固定版本确认后再选用。GQA、pooling/tail、跨层共享 cache 或其他运行时元数据需要新的适配规则，不能只替换维度后自动继承。当前公开模型入口与虚构案例只证明生成流程和这些计算规则的适用范围。

## 实现位置

- `lib/model-memory/engine.mjs`：局部 shape、packed weight/scale、EP 槽位、PP 归属、cache 和 buffer 数值计算。
- `lib/model-memory/schema.mjs`：模型描述检查及显式量化配方选择。
- `lib/model-memory/kimi.mjs`：仓库已有公开配置的模型适配。
- `scripts/memory-workbook.mjs`：统一工作簿、原生公式、输入和排版。
- `scripts/verify-memory-sheet.mjs`：与数值引擎对照的联动及边界检查。
- `tests/model-memory.test.mjs`：公开配置回归和人工可核算的虚构小规格测试。
