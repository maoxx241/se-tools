import { strict as assert } from "node:assert";
import fs from "node:fs/promises";
import path from "node:path";

import { models, modelSlugs, scenarioRow } from "./build-analysis-workbook.mjs";
import { elementCount, modelSpec, shapeText } from "./model-analysis-specs.mjs";
import { loadSpreadsheetRuntime } from "../lib/spreadsheet-runtime.mjs";

const { FileBlob, SpreadsheetFile } = await loadSpreadsheetRuntime();

const inputRoot = process.argv[2] || path.resolve("outputs/models");
const selected = process.env.MODEL
  ? models.filter((model) => model.key === process.env.MODEL || modelSlugs[model.key] === process.env.MODEL)
  : models;
if (!selected.length) throw new Error(`Unknown MODEL: ${process.env.MODEL}`);

const results = [];

for (const model of selected) {
  const slug = modelSlugs[model.key];
  const modelDir = path.join(inputRoot, slug);
  const inputPath = path.join(modelDir, `${slug}-analysis.xlsx`);
  const config = JSON.parse(await fs.readFile(path.join(modelDir, "config.json"), "utf8"));
  const spec = modelSpec(model, config);
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));

  for (const phase of ["Prefill", "Decode"]) {
    const sheet = workbook.worksheets.getItem(phase);
    const matrix = sheet.getRange("A1:P260").values;
    assert.deepEqual(matrix[0].slice(0, 5), ["模块", "权重 / Tensor", "shape", "dtype", "字节/元素"]);

    for (let index = 0; index < spec.rows.length; index++) {
      const actual = matrix[index + 1];
      const expected = spec.rows[index];
      assert.equal(actual[0], expected.module);
      assert.equal(actual[1], expected.name);
      assert.equal(actual[2], shapeText(expected.dims));
      assert.equal(actual[3], expected.dtype);
      assert.equal(actual[4], expected.bytes);
      assert.equal(actual[5], expected.count);
      assert.equal(actual[6], elementCount(expected.dims));
    }

    const communicationHeader = matrix.findIndex((r) => r[0] === "并行策略" && r[3] === "Collective");
    assert.ok(communicationHeader > spec.rows.length);
    const communicationRows = matrix.slice(communicationHeader + 1).filter((r) => r[0] && !["合计", "单 Rank 事件合计"].includes(r[0]));
    const strategies = new Set(communicationRows.map((r) => r[0]));
    assert.ok(strategies.has("SP"));
    assert.ok(strategies.has("LM Head TP"));
    if (spec.facts.moeLayers > 0) {
      assert.ok(strategies.has("EP"));
      assert.ok(strategies.has("DP"));
      assert.ok(strategies.has("共享专家 TP"));
    }
    if (["deepseek_v4", "glm53"].includes(model.profile)) assert.ok(strategies.has("DSA CP"));
    if (phase === "Prefill") assert.ok(strategies.has("PCP") || spec.facts.moeLayers === 0);
    if (phase === "Decode" && spec.facts.dcpAttentionLayers > 0 && !["deepseek_v4", "glm53"].includes(model.profile)) assert.ok(strategies.has("DCP"));

    const expectedStep = phase === "Prefill" ? 8192 : Math.min(8192, 128 * (model.defaultSpecSteps + 1));
    assert.equal(sheet.getRange(`P${scenarioRow.stepTokens}`).values[0][0], expectedStep);
    assert.equal(sheet.getRange(`P${scenarioRow.tp}`).values[0][0], 8);
    assert.equal(sheet.getRange(`P${scenarioRow.sp}`).values[0][0], 1);
    if (phase === "Prefill") assert.equal(sheet.getRange(`P${scenarioRow.pcp}`).values[0][0], 1);
    else assert.ok([null, ""].includes(sheet.getRange(`P${scenarioRow.pcp}`).values[0][0]));
    if (phase === "Decode" && !["deepseek_v4", "glm53"].includes(model.profile)) assert.equal(sheet.getRange(`P${scenarioRow.dcp}`).values[0][0], 1);
    else assert.ok([null, ""].includes(sheet.getRange(`P${scenarioRow.dcp}`).values[0][0]));

    const qAProj = spec.rows.filter((r) => /q_a_proj\.weight$/.test(r.name));
    const qANorm = spec.rows.filter((r) => /q_a_(?:layer)?norm\.weight$/.test(r.name));
    if (qAProj.length && qANorm.length) assert.ok(qAProj.every((r) => !r.name.includes("+")));

    const firstWeightGiB = sheet.getRange("J2").values[0][0];
    const originalBytes = sheet.getRange("E2").values[0][0];
    sheet.getRange("E2").values = [[originalBytes / 2]];
    assert.ok(Math.abs(sheet.getRange("J2").values[0][0] * 2 - firstWeightGiB) < 1e-9);

    const activeComm = communicationRows.find((r) => Number(r[7]) > 0 && Number(r[9]) > 0);
    assert.ok(activeComm);
    const activeIndex = matrix.findIndex((r) => r === activeComm);
    const originalActBytes = sheet.getRange(`P${scenarioRow.activationBytes}`).values[0][0];
    const originalSent = sheet.getRange(`J${activeIndex + 1}`).values[0][0];
    sheet.getRange(`P${scenarioRow.activationBytes}`).values = [[originalActBytes / 2]];
    const changedSent = sheet.getRange(`J${activeIndex + 1}`).values[0][0];
    if (activeComm[5] === originalActBytes) assert.ok(Math.abs(changedSent * 2 - originalSent) < 1e-8);

    const forbidden = await workbook.inspect({
      kind: "match",
      searchTerm: "Safetensors|Checkpoint Tensor|Device工作集|Host规划|手动Kernel Workspace|q_a_proj \\+ q_a_norm|逻辑权重|并行轴|适用层|使用层|来源|兜底|猜测",
      options: { useRegex: true, maxResults: 100 },
      maxChars: 10000,
    });
    assert.match(forbidden.ndjson, /matched 0 entries/);
  }

  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 200 },
    maxChars: 20000,
  });
  assert.match(errors.ndjson, /matched 0 entries/);
  await fs.rm(`${inputPath}.inspect.ndjson`, { force: true });
  results.push({ model: model.key, inputPath, weightRows: spec.rows.length, formulaErrors: 0 });
}

console.log(JSON.stringify(results, null, 2));
