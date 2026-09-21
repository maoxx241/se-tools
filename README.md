# AI Infra 模型分析工具

用模型 `config.json` 和实现代码分析权重、权重切分及通信量，结果可在 Excel 中直接修改和复算。本仓库收录了 2026-09-04 的分析结果，并用 [CANN HCCL](https://gitcode.com/cann/hccl) 的 API、算法文档和实现补充了通信量口径。

## 从这里开始

- **看结论与场景**：[分析记录](analysis/SESSION.md)。包含范围、模型差异、最终 P/D 拓扑和保留的问题。
- **Engram offloading / 前缀池化**：[两页 H2D / RH2D Excel](examples/host-transfers/)、[公式与源码](analysis/HOST-TRANSFERS.md)。V4.1 token lookup 搬运；六模型 EP32/256、100% 前缀命中的 D host→P device 数据量，输入可编辑。
- **PD KV 抢带宽对耗时的影响**：[源码分析与可编辑带宽说明](analysis/PD-CONTENTION.md)、[一分钟窗口场景](analysis/PD-MINUTE.md)、[数值示例](examples/pd-contention/)。新增 144 个 TPOT/TTFT/长度/EP 组合，分列部分与全 EP 重叠次数及耗时增量；4800 GB/s 示例可按模型、EP 分别修改，EP 与 KV 带宽独立。原手动覆盖率区域保留。
- **DeepSeek V4.1 Flash**：[模型及 Excel](models/deepseek-v4.1-flash/)、[VA main 实现分析](analysis/DEEPSEEK-V4.1.md)。独立处理跨层缓存、DSA CP、Engram 和源权重/驻留格式差异。
- **EP32 / EP256 新规格**：[六模型通信与 KV 存取对比](examples/kv-ep-sweep/)、[公式和源码依据](analysis/KV-EP-SPECS.md)。包含 V4.1 Flash；随 DP 域扩大刷新 Prefill AllToAllV、Decode MC2，分列 Prefill 生成、保留和 P→D 拉取量。V4.1 拉取是固定版本的规划值；新 main 的 PD 可用性未在本次实机验证。
- **直接使用 Excel**：[模型目录](models/)内每个模型各有一份 Prefill / Decode 工作簿；[256K 通信需求案例](examples/communication-256k/)包含跨模型结果和复算输出。
- **看计算依据**：[权重与通信方法](analysis/METHODOLOGY.md)、[HCCL 校正](analysis/HCCL.md)、[来源版本](analysis/EVIDENCE.md)。
- **接入新模型或 Profiling**：[扩展与复算指南](analysis/REPRODUCING.md)。

原有分析来自 2026-09-04；V4.1 是 2026-09-12 新增分析。2026-09-20 在当前 Excel 下方追加 PD 耗时区域，旧数据保持不变，原始字节副本移至[归档目录](examples/archive/pre-pd-contention-20260920/)。config 均单独保存。逐模型 Excel 的右侧是可编辑参数，权重表中的字节/元素也可以直接修改；历史通信需求表的 I 列是固定场景文本，不是公式驱动的通用模板。两类表的默认参数不同。

## 无需 Codex 的复算

需要 Node.js 20+；核心计算没有第三方依赖，不下载权重、不联网、不需要 NPU。

```bash
git clone https://github.com/maoxx241/se-tools.git
cd se-tools
npm test
node scripts/export-analysis.mjs weights kimi-k3
node scripts/export-analysis.mjs communication
node scripts/export-analysis.mjs kv-ep
node scripts/export-analysis.mjs contention
node scripts/export-analysis.mjs host-transfers
node scripts/export-analysis.mjs weights deepseek-v4.1-flash
node scripts/export-analysis.mjs communication deepseek-v4.1-flash
node scripts/export-analysis.mjs engram examples/hccl/engram.json
node scripts/export-analysis.mjs collective examples/hccl/allreduce.json
node scripts/export-analysis.mjs collective examples/hccl/alltoallv.json
python3 scripts/verify-archive.py
python3 scripts/verify-kv-ep.py
python3 scripts/verify-contention.py
```

`weights` 输出 Tensor 形状、dtype、总字节数和切分规则；不带模型参数的 `communication` 复算历史六模型场景，指定 V4.1 则输出新增案例；`collective` 按 HCCL API count 口径计算；`engram` 计算查询和反向响应的不均衡流量。JSON 中大字节数用十进制字符串表示。

## 内容

```text
models/<model>/                    config.json、历史 Excel、模型说明
examples/communication-256k/       最终 P/D 场景、历史 Excel、复算数值
examples/kv-ep-sweep/              EP32/256、DP 域变化、KV 存取公式及 Excel
examples/pd-contention/            可编辑有效带宽、PD 争用毫秒估算及说明
examples/hccl/                     API count 与非均衡 AlltoAllV 示例
lib/collectives.mjs                HCCL count、Ring、AlltoAllV 公共计算
lib/communication-requirements.mjs  256K 场景与 P→D 缓存计算
lib/kv-ep-specs.mjs                六模型逐组件 KV 与 AllToAllV / MC2 估算
lib/pd-contention.mjs             带宽降幅到通信/forward 增量的公共计算
lib/model-catalog.mjs              模型与来源 revision
scripts/model-analysis-specs.mjs   模型专属 Tensor 清单
scripts/                          导出、可选 Excel 构建及验证
analysis/                         结论、方法、证据与扩展指南
```

## 重新生成 Excel

已有 Excel 用 Excel 或 LibreOffice 即可编辑。重新生成 `.xlsx` 的原始脚本需要额外提供 `@oai/artifact-tool` 运行环境；它不是核心工具的 npm 安装依赖。环境中已有该包时可直接运行，或将 `ARTIFACT_TOOL_ENTRY` 设置为该包的绝对模块入口：

```bash
node scripts/build-analysis-workbook.mjs outputs/models
node scripts/verify-analysis-workbook.mjs outputs/models
node scripts/build-kv-ep-workbook.mjs outputs/kv-ep
node scripts/build-host-transfer-workbook.mjs outputs/host-transfers
MODEL=kimi-k3 node scripts/build-analysis-workbook.mjs outputs/models
# 使用已有通信需求表布局重新填值；仅写六个模型的 I4:I18
node scripts/fill-communication-requirement.mjs \
  examples/communication-256k/communication-requirements-20260904.xlsx \
  outputs/communication-requirements.xlsx
```

配置默认来自仓库 `models/`，可通过 `MODEL_CONFIG_ROOT` 显式指定其他模型目录。生成件写入被 Git 忽略的 `outputs/`，归档 Excel 不随默认构建覆盖。

## 当前边界

已实现权重建模、切分公式、通信事件清单、Ring 基准和 AlltoAllV 端点数据量。HCCL 算法选择、分层链路流量、协议开销及实测耗时需要运行时证据，不能从 Ring 系数直接得到。MFU/MBU、完整访存/算子性能仿真尚未实现。版本与限制见[来源记录](analysis/EVIDENCE.md)。
