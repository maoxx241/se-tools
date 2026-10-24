# 通用显存分析示例

[synthetic.json](synthetic.json) 是人为构造的六层小模型，用于演示模型描述和回归测试。它没有对应的真实权重或部署，不代表任何实际模型规格。

已生成的[四页工作簿](synthetic-hybrid/synthetic-hybrid-memory.xlsx)、[数值结果](synthetic-hybrid/memory-results.json)和[验证记录](synthetic-hybrid/memory-verification.json)可直接查看。

```bash
node scripts/export-analysis.mjs memory --spec examples/model-memory/synthetic.json
node scripts/build-memory-workbook.mjs --spec examples/model-memory/synthetic.json --render
```

示例分开声明基础 Tensor、量化配方、P/D 参数和 PP 分层。第一段 PP 刻意不包含 MLA 层，以检查其他 stage 的 buffer 是否仍正确计入。输入输出及扩展方法见[模型显存工作簿说明](../../analysis/MODEL-MEMORY.md)。
