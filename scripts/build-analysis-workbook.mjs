import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { elementCount, modelSpec, shapeText } from "./model-analysis-specs.mjs";
import { loadSpreadsheetRuntime } from "../lib/spreadsheet-runtime.mjs";
import { v41WorkbookEvents, v41WorkbookNotes } from './deepseek-v41-workbook.mjs';

const { SpreadsheetFile, Workbook } = await loadSpreadsheetRuntime();

const GiB = 1073741824;
const MiB = 1048576;
import { models, modelSlugs, readModelConfig } from "../lib/model-catalog.mjs";
import { sendRatioFormula } from "../lib/collectives.mjs";
export { models, modelSlugs, VLLM_REVISION, VLLM_ASCEND_REVISION } from "../lib/model-catalog.mjs";

const scenarioGroups = [
  ["请求参数", [
    ["maxBatchTokens", "max_num_batched_tokens"], ["maxSeqs", "max_num_seqs"],
    ["inputTokens", "平均输入 Token 数"], ["outputTokens", "平均输出 Token 数"],
    ["specSteps", "投机采样步数"], ["stepTokens", "当前 Step Token 数"], ["samplingRows", "本 Step 采样行数"],
  ]],
  ["并行策略", [
    ["dp", "DP"], ["tp", "TP"], ["sp", "SP（0/1）"], ["ep", "EP"],
    ["pcp", "PCP（1 表示关闭）"], ["dcp", "DCP（1 表示关闭）"],
    ["sharedMode", "共享专家并行方式"], ["sharedTP", "共享专家 TP"],
    ["otp", "O Proj TP（0 表示使用 TP）"], ["lmheadTP", "LM Head TP（0 表示使用 TP）"],
    ["embeddingTP", "Embedding TP（0 表示使用 TP）"],
    ["fineOTP", "Fine O TP（0/1）"], ["fineLMHead", "Fine LM Head TP（0/1）"],
    ["fineEmbedding", "Fine Embedding TP（0/1）"], ["dsaCP", "DSA CP（0/1）"],
  ]],
  ["通信参数", [
    ["activationBytes", "激活字节/元素"], ["lseBytes", "LSE 字节/元素"],
    ["tokenIdBytes", "Token ID 字节/元素"], ["indexBytes", "Index 字节/元素"],
    ["routeFraction", "跨 EP Rank 路由比例"], ["reduceSample", "Reduce Sample（0/1）"],
    ["candidateK", "Reduce Sample K"],
  ]],
  ["V4.1 节点与载荷", [
    ["nodeRanks", "每节点 Rank 数"], ["engramEnabled", "Engram（0/1）"],
    ["dispatchBytes", "MoE Dispatch 字节/元素"], ["combineBytes", "MoE Combine 字节/元素"],
    ["cacheRequests", "缓存案例请求数"],
  ]],
];

export const scenarioRow = {};
const scenarioGroupRow = {};
let nextScenarioRow = 1;
for (const [title, items] of scenarioGroups) {
  scenarioGroupRow[title] = nextScenarioRow++;
  for (const [key] of items) scenarioRow[key] = nextScenarioRow++;
  nextScenarioRow++;
}

function sc(phase, key) {
  return `'${phase}'!$P$${scenarioRow[key]}`;
}

async function loadPinnedConfig(model, modelDir) {
  const config = await readModelConfig(modelSlugs[model.key]);
  await fs.writeFile(path.join(modelDir, "config.json"), `${JSON.stringify(config, null, 2)}\n`);
  return config;
}

function activeScenarioGroups(model, phase) {
  if (model.profile === 'deepseek_v41') {
    const keys = new Set(['maxBatchTokens', 'maxSeqs', 'inputTokens', 'outputTokens', 'specSteps', 'stepTokens', 'samplingRows', 'dp', 'tp', 'sp', 'ep', 'dsaCP', 'activationBytes', 'routeFraction', 'nodeRanks', 'engramEnabled', 'dispatchBytes', 'combineBytes', 'cacheRequests']);
    return scenarioGroups.map(([title, items]) => [title, items.filter(([key]) => keys.has(key))]);
  }
  const dsa = ["deepseek_v4", "glm53"].includes(model.profile);
  return scenarioGroups.filter(([title]) => title !== 'V4.1 节点与载荷').map(([title, items]) => [title, items.filter(([key]) => {
    if (key === "dsaCP") return dsa;
    if (key === "pcp") return phase === "Prefill";
    if (key === "dcp") return phase === "Decode" && !dsa;
    return true;
  })]);
}

function effectiveGroup(phase, key) {
  const S = (name) => sc(phase, name);
  if (key === "embedding") return `IF(${S("embeddingTP")}>0,${S("embeddingTP")},${S("tp")})`;
  if (key === "lmhead") return `IF(${S("lmheadTP")}>0,${S("lmheadTP")},${S("tp")})`;
  if (key === "otp") return `IF(${S("otp")}>0,${S("otp")},${S("tp")})`;
  if (key === "shared") return `IF(${S("sharedMode")}="TP",${S("sharedTP")},1)`;
  return S(key);
}

function partitionLabel(key) {
  return {
    replicated: "Replicated", tp: "TP", ep_individual: "EP", ep_tensor: "EP",
    shared: "共享专家 TP/DP", otp: "O Proj TP", embedding: "Embedding TP",
    lmhead: "LM Head TP", dsa_tp: "DSA CP / TP",
    node_shard: "节点内均摊（源格式）", v41_shared: "SP 复制 / 非 SP TP",
  }[key];
}

function partitionCountFormula(phase, key) {
  const S = (name) => sc(phase, name);
  if (key === 'node_shard') return `=${S('nodeRanks')}`;
  if (key === 'v41_shared') return `=IF(${S('sp')}<>0,1,${S('tp')})`;
  if (key === "tp") return `=${S("tp")}`;
  if (key === "ep_individual" || key === "ep_tensor") return `=${S("ep")}`;
  if (key === "shared") return `=${effectiveGroup(phase, "shared")}`;
  if (key === "otp") return `=${effectiveGroup(phase, "otp")}`;
  if (key === "embedding") return `=${effectiveGroup(phase, "embedding")}`;
  if (key === "lmhead") return `=${effectiveGroup(phase, "lmhead")}`;
  if (key === "dsa_tp") return `=IF(${S("dsaCP")}<>0,1,${S("tp")})`;
  return "=1";
}

function styleHeader(sheet, row, lastColumn) {
  sheet.getRange(`A${row}:${lastColumn}${row}`).format = {
    fill: "#D9E2F3", font: { name: "Arial", size: 10, bold: true, color: "#000000" },
    borders: { preset: "all", style: "thin", color: "#A6A6A6" },
    horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, rowHeight: 30,
  };
}

function setWidths(sheet, widths) {
  for (const [range, width] of widths) sheet.getRange(range).format.columnWidth = width;
}

function buildScenario(sheet, model, phase) {
  const groups = activeScenarioGroups(model, phase);
  const dsa = ["deepseek_v4", "glm53"].includes(model.profile);
  const values = {
    maxBatchTokens: 8192, maxSeqs: 128, inputTokens: 4096, outputTokens: 1024,
    specSteps: phase === "Decode" ? model.defaultSpecSteps : 0, stepTokens: null, samplingRows: null,
    dp: 1, tp: 8, sp: 1, ep: model.defaultEP, pcp: 1, dcp: 1,
    sharedMode: "TP", sharedTP: 8, otp: 0, lmheadTP: 0, embeddingTP: 0,
    fineOTP: 0, fineLMHead: 0, fineEmbedding: 0, dsaCP: dsa ? 1 : 0,
    activationBytes: 2, lseBytes: 4, tokenIdBytes: 8, indexBytes: 4,
    routeFraction: null, reduceSample: 1, candidateK: 256,
    nodeRanks: 8, engramEnabled: 1, dispatchBytes: 2, combineBytes: 2, cacheRequests: 16,
  };
  if (model.profile === 'deepseek_v41') Object.assign(values, {
    maxBatchTokens: phase === 'Prefill' ? 16384 : 128, maxSeqs: 128, inputTokens: 262144,
    tp: phase === 'Prefill' ? 8 : 1, dp: phase === 'Prefill' ? 4 : 32,
    sp: phase === 'Prefill' ? 1 : 0, dsaCP: phase === 'Prefill' ? 1 : 0,
  });
  for (const [title, items] of groups) {
    const titleRow = scenarioGroupRow[title];
    sheet.getRange(`O${titleRow}:P${titleRow}`).values = [[title, null]];
    sheet.getRange(`O${titleRow}:P${titleRow}`).format = {
      fill: "#D9E2F3", font: { name: "Arial", size: 10, bold: true, color: "#000000" },
      borders: { preset: "all", style: "thin", color: "#A6A6A6" },
    };
    for (const [key, label] of items) sheet.getRange(`O${scenarioRow[key]}:P${scenarioRow[key]}`).values = [[label, values[key]]];
  }
  const c = (key) => `$P$${scenarioRow[key]}`;
  sheet.getRange(`P${scenarioRow.stepTokens}`).formulas = [[phase === "Prefill"
    ? `=MIN(${c("maxBatchTokens")},${c("maxSeqs")}*${c("inputTokens")})`
    : `=MIN(${c("maxBatchTokens")},${c("maxSeqs")}*(${c("specSteps")}+1))`]];
  sheet.getRange(`P${scenarioRow.samplingRows}`).formulas = [[phase === "Prefill"
    ? `=MIN(${c("maxSeqs")},${c("stepTokens")})`
    : `=MIN(${c("stepTokens")},${c("maxSeqs")}*(${c("specSteps")}+1))`]];
  sheet.getRange(`P${scenarioRow.routeFraction}`).formulas = [[`=IF(${c("ep")}>1,(${c("ep")}-1)/${c("ep")},0)`]];
  for (const [, items] of groups) {
    const first = scenarioRow[items[0][0]];
    const last = scenarioRow[items.at(-1)[0]];
    sheet.getRange(`O${first}:P${last}`).format.borders = { preset: "all", style: "thin", color: "#D9D9D9" };
    for (const [key] of items) {
      const cell = sheet.getRange(`P${scenarioRow[key]}`);
      cell.format.fill = ["stepTokens", "samplingRows", "routeFraction"].includes(key) ? "#F2F2F2" : "#FFF2CC";
      if (key !== "sharedMode") cell.setNumberFormat(key === "routeFraction" ? "0.0000" : "0.##");
    }
  }
}

function buildWeightTable(sheet, rows, phase, startRow) {
  const headers = ["模块", "权重 / Tensor", "shape", "dtype", "字节/元素", "数量", "单个 Tensor 元素数", "切分方式", "切分份数", "总权重 GiB", "单 Rank 权重 GiB"];
  sheet.getRange(`A${startRow}:K${startRow + rows.length}`).values = [headers, ...rows.map((r) => [
    r.module, r.name, shapeText(r.dims), r.dtype, r.bytes, r.count, elementCount(r.dims), partitionLabel(r.partition), null, null, null,
  ])];
  const first = startRow + 1;
  for (let i = 0; i < rows.length; i++) {
    const excelRow = first + i;
    sheet.getRange(`I${excelRow}`).formulas = [[partitionCountFormula(phase, rows[i].partition)]];
    sheet.getRange(`J${excelRow}`).formulas = [[`=E${excelRow}*F${excelRow}*G${excelRow}/${GiB}`]];
    sheet.getRange(`K${excelRow}`).formulas = [[`=J${excelRow}/I${excelRow}`]];
  }
  const totalRow = first + rows.length;
  sheet.getRange(`A${totalRow}:K${totalRow}`).values = [["合计", null, null, null, null, null, null, null, null, null, null]];
  sheet.getRange(`J${totalRow}`).formulas = [[`=SUM(J${first}:J${totalRow - 1})`]];
  sheet.getRange(`K${totalRow}`).formulas = [[`=SUM(K${first}:K${totalRow - 1})`]];
  sheet.getRange(`A${totalRow}:K${totalRow}`).format.font = { bold: true };
  sheet.getRange(`E${first}:E${totalRow - 1}`).format.fill = "#FFF2CC";
  sheet.getRange(`E${first}:I${totalRow - 1}`).setNumberFormat("#,##0.####");
  sheet.getRange(`J${first}:K${totalRow}`).setNumberFormat("0.000000");
  styleHeader(sheet, startRow, "K");
  return { first, last: totalRow };
}

function communicationEvents(model, phase, facts) {
  const S = (key) => sc(phase, key);
  if (model.profile === 'deepseek_v41') return v41WorkbookEvents(S, facts);
  const T = S("stepTokens");
  const tPad = `(ROUNDUP(${T}/${S("tp")},0)*${S("tp")})`;
  const tLocal = `(${tPad}/${S("tp")})`;
  const pcpPad = `(ROUNDUP(${T}/${S("pcp")},0)*${S("pcp")})`;
  const pcpLocal = `(${pcpPad}/${S("pcp")})`;
  const localForMoe = `IF(${S("sp")}<>0,${tLocal},${T})`;
  const act = S("activationBytes");
  const events = [];
  const add = (strategy, module, location, collective, elements, bytes, group, count) => events.push({ strategy, module, location, collective, elements: `=${elements}`, bytes: `=${bytes}`, group: `=${group}`, count: `=${count}` });
  const embeddingGroup = effectiveGroup(phase, "embedding");
  const lmGroup = effectiveGroup(phase, "lmhead");
  const oGroup = effectiveGroup(phase, "otp");
  const sharedGroup = effectiveGroup(phase, "shared");

  if (model.profile !== "kimi_dspark") {
    add("Embedding TP", "Embedding", "标准 TP 输出", "Ring AllReduce", `${T}*${facts.H}`, act, embeddingGroup, `IF(AND(${embeddingGroup}>1,${S("fineEmbedding")}=0),1,0)`);
    add("Embedding TP", "Embedding", "Fine TP Token ID", "Ring AllGather", T, S("tokenIdBytes"), embeddingGroup, `IF(AND(${embeddingGroup}>1,${S("fineEmbedding")}<>0),1,0)`);
    add("Embedding TP", "Embedding", "Fine TP Hidden", "Ring ReduceScatter", `${T}*${embeddingGroup}*${facts.H}`, act, embeddingGroup, `IF(AND(${embeddingGroup}>1,${S("fineEmbedding")}<>0),1,0)`);
  } else {
    add("TP", "Input Projection", "Column Parallel 输出", "Ring AllGather", `${T}*${facts.H}/${S("tp")}`, act, S("tp"), `IF(${S("tp")}>1,1,0)`);
  }

  if (!["deepseek_v4", "glm53"].includes(model.profile)) {
    const layerCount = facts.tpAttentionLayers;
    add("SP", "Attention", "Attention 前", "Ring AllGather", `${tLocal}*${facts.H}`, act, S("tp"), `${layerCount}*IF(AND(${S("sp")}<>0,${S("tp")}>1,${S("fineOTP")}=0),1,0)`);
    add("SP", "Attention", "O Proj 后", "Ring ReduceScatter", `${tPad}*${facts.H}`, act, S("tp"), `${layerCount}*IF(AND(${S("sp")}<>0,${S("tp")}>1,${S("fineOTP")}=0),1,0)`);
    add("O Proj TP", "Attention", "非 SP 标准 TP", "Ring AllReduce", `${T}*${facts.H}`, act, oGroup, `${layerCount}*IF(AND(${S("sp")}=0,${oGroup}>1,${S("fineOTP")}=0),1,0)`);
    add("Fine O TP", "Attention", "O Proj 输入", "Equal-split AllToAll", `${T}*${facts.heads * facts.headDim}`, act, oGroup, `${layerCount}*IF(AND(${oGroup}>1,${S("fineOTP")}<>0),1,0)`);
    add("Fine O TP", "Attention", "O Proj 输出", "Ring ReduceScatter", `${T}*${oGroup}*${facts.H}`, act, oGroup, `${layerCount}*IF(AND(${oGroup}>1,${S("fineOTP")}<>0),1,0)`);
  } else {
    add("DSA CP", "Sparse Attention", "Q/KV 投影输入", "Ring AllGather", `${tLocal}*${facts.H}`, act, S("tp"), `${facts.attentionLayers}*IF(AND(${S("dsaCP")}<>0,${S("tp")}>1),1,0)`);
    add("DSA CP", "Sparse Attention", "恢复 TP Head Layout", "Equal-split AllToAll", `${tLocal}*${facts.heads}*${facts.headDim}`, act, S("tp"), `${facts.attentionLayers}*IF(AND(${S("dsaCP")}<>0,${S("tp")}>1),1,0)`);
    if (facts.indexerLayers > 0) add("DSA CP", "Indexer", "Top-k Index", "Ring AllGather", `${tLocal}*${facts.indexTopK}`, S("indexBytes"), S("tp"), `${facts.indexerLayers}*IF(AND(${S("dsaCP")}<>0,${S("tp")}>1),1,0)`);
    add("O Proj TP", "Sparse Attention", "DSA CP 关闭", "Ring AllReduce", `${T}*${facts.H}`, act, oGroup, `${facts.attentionLayers}*IF(AND(${S("dsaCP")}=0,${oGroup}>1),1,0)`);
  }

  if (facts.denseLayers > 0) add("SP", "Dense MLP", "Down Proj 输出", "Ring AllReduce", `${localForMoe}*${facts.H}`, act, S("tp"), `${facts.denseLayers}*IF(${S("tp")}>1,1,0)`);

  if (facts.moeLayers > 0) {
    add("EP", "Routed Experts", "Dispatch Hidden", "AllToAllV（跨 Rank 发送）", `${localForMoe}*${facts.topK}*${S("routeFraction")}*${facts.H}`, act, S("ep"), `${facts.moeLayers}*IF(${S("ep")}>1,1,0)`);
    add("EP", "Routed Experts", "Combine Hidden", "AllToAllV（跨 Rank 发送）", `${localForMoe}*${facts.topK}*${S("routeFraction")}*${facts.H}`, act, S("ep"), `${facts.moeLayers}*IF(${S("ep")}>1,1,0)`);
    add("DP", "MoE", "Hidden 输入", "Ring AllGather", `${T}*${facts.H}`, act, S("dp"), `${facts.moeLayers}*IF(AND(${S("dp")}>1,${S("ep")}=1),1,0)`);
    add("DP", "MoE Router", "Router Logits", "Ring AllGather", `${T}*${facts.experts}`, act, S("dp"), `${facts.moeLayers}*IF(AND(${S("dp")}>1,${S("ep")}=1),1,0)`);
    add("DP", "MoE", "Routed Expert 输出", "Ring ReduceScatter", `${T}*${S("dp")}*${facts.H}`, act, S("dp"), `${facts.moeLayers}*IF(AND(${S("dp")}>1,${S("ep")}=1),1,0)`);
    add("共享专家 TP", "Shared Experts", "SP 输入", "Ring AllGather", `${tLocal}*${facts.H}`, act, sharedGroup, `${facts.moeLayers}*IF(AND(${S("sharedMode")}="TP",${sharedGroup}>1,${S("sp")}<>0),1,0)`);
    add("共享专家 TP", "Shared Experts", "SP 输出", "Ring ReduceScatter", `${tPad}*${facts.H}`, act, sharedGroup, `${facts.moeLayers}*IF(AND(${S("sharedMode")}="TP",${sharedGroup}>1,${S("sp")}<>0),1,0)`);
    add("共享专家 TP", "Shared Experts", "非 SP 输出", "Ring AllReduce", `${T}*${facts.H}`, act, sharedGroup, `${facts.moeLayers}*IF(AND(${S("sharedMode")}="TP",${sharedGroup}>1,${S("sp")}=0),1,0)`);
    if (phase === "Prefill") {
      add("PCP", "MoE", "Hidden 输入", "Ring AllGather", `${pcpLocal}*${facts.H}`, act, S("pcp"), `${facts.moeLayers}*IF(${S("pcp")}>1,1,0)`);
      add("PCP", "MoE Router", "Router Logits", "Ring AllGather", `${pcpLocal}*${facts.experts}`, act, S("pcp"), `${facts.moeLayers}*IF(${S("pcp")}>1,1,0)`);
      add("PCP", "MoE", "Routed Expert 输出", "Ring ReduceScatter", `${pcpPad}*${facts.H}`, act, S("pcp"), `${facts.moeLayers}*IF(${S("pcp")}>1,1,0)`);
    }
  }

  if (phase === "Prefill") {
    if (facts.kvRank > 0) add("PCP", "Attention KV", "KV Cache 写入前", "Ring AllGather", `${pcpLocal}*${facts.kvRank + facts.ropeDim}`, act, S("pcp"), `${facts.attentionLayers}*IF(${S("pcp")}>1,1,0)`);
    else if (facts.dcpAttentionLayers > 0) add("PCP", "Attention KV", "KV Cache 写入前", "Ring AllGather", `${pcpLocal}*2*MAX(1,${facts.kvHeads}/${S("tp")})*${facts.headDim}`, act, S("pcp"), `${facts.dcpAttentionLayers}*IF(${S("pcp")}>1,1,0)`);
  }

  if (phase === "Decode" && facts.dcpAttentionLayers > 0 && !["deepseek_v4", "glm53"].includes(model.profile)) {
    const localHeads = `(${facts.heads}/${S("tp")})`;
    const gatheredHeads = `(${facts.heads}*${S("dcp")}/${S("tp")})`;
    add("DCP", "Attention", "Query", "Ring AllGather", `${T}*${localHeads}*${facts.headDim}`, act, S("dcp"), `${facts.dcpAttentionLayers}*IF(${S("dcp")}>1,1,0)`);
    add("DCP", "Attention", "LSE", "Ring AllGather", `${T}*${gatheredHeads}`, S("lseBytes"), S("dcp"), `${facts.dcpAttentionLayers}*IF(${S("dcp")}>1,1,0)`);
    add("DCP", "Attention", "Attention 输出", "Ring ReduceScatter", `${T}*${gatheredHeads}*${facts.headDim}`, act, S("dcp"), `${facts.dcpAttentionLayers}*IF(${S("dcp")}>1,1,0)`);
  }

  add("SP", "Model Output", "Final Norm 后", "Ring AllGather", `${tLocal}*${facts.H}`, act, S("tp"), `IF(AND(${S("sp")}<>0,${S("tp")}>1),1,0)`);
  add("LM Head TP", "LM Head", "完整 Logits", "Ring AllGather", `${S("samplingRows")}*${facts.vocab}/${lmGroup}`, act, lmGroup, `IF(AND(${lmGroup}>1,${S("fineLMHead")}=0,${S("reduceSample")}=0),1,0)`);
  add("LM Head TP", "LM Head", "Reduce Sample", "Ring AllGather", `${S("samplingRows")}*${S("candidateK")}`, `(${act}+${S("indexBytes")})`, lmGroup, `IF(AND(${lmGroup}>1,${S("fineLMHead")}=0,${S("reduceSample")}<>0),1,0)`);
  add("Fine LM Head TP", "LM Head", "Hidden 输入", "Ring AllGather", `${S("samplingRows")}*${facts.H}`, act, lmGroup, `IF(AND(${lmGroup}>1,${S("fineLMHead")}<>0),1,0)`);
  add("Fine LM Head TP", "LM Head", "完整 Logits", "Equal-split AllToAll", `${S("samplingRows")}*${facts.vocab}`, act, lmGroup, `IF(AND(${lmGroup}>1,${S("fineLMHead")}<>0,${S("reduceSample")}=0),1,0)`);
  add("Fine LM Head TP", "LM Head", "Reduce Sample", "Ring AllGather", `${S("samplingRows")}*${S("candidateK")}`, `(${act}+${S("indexBytes")})`, lmGroup, `IF(AND(${lmGroup}>1,${S("fineLMHead")}<>0,${S("reduceSample")}<>0),1,0)`);
  return events;
}

function buildCommunicationTable(sheet, model, phase, facts, startRow) {
  const events = communicationEvents(model, phase, facts);
  const headers = ["并行策略", "模块", "通信位置", "Collective", "本 Rank 输入元素/次", "字节/元素", "通信组大小", "次数/Step", "本 Rank 载荷 MiB/次", "单 Rank 建模发送 MiB/次", "单 Rank 建模发送 MiB/Step", "通信组建模发送 MiB/Step"];
  if (model.profile === 'deepseek_v41') {
    headers[4] = '输入/跨 Rank 期望元素';
    headers[9] = 'Rank 平均发送 MiB/次';
    headers[10] = 'Rank 平均发送 MiB/Step';
  }
  sheet.getRange(`A${startRow}:L${startRow + events.length}`).values = [headers, ...events.map((e) => [e.strategy, e.module, e.location, e.collective, null, null, null, null, null, null, null, null])];
  const first = startRow + 1;
  for (let i = 0; i < events.length; i++) {
    const r = first + i;
    const e = events[i];
    sheet.getRange(`E${r}`).formulas = [[e.elements]];
    sheet.getRange(`F${r}`).formulas = [[e.bytes]];
    sheet.getRange(`G${r}`).formulas = [[e.group]];
    sheet.getRange(`H${r}`).formulas = [[e.count]];
    sheet.getRange(`I${r}`).formulas = [[`=E${r}*F${r}/${MiB}`]];
    sheet.getRange(`J${r}`).formulas = [[`=I${r}*(${sendRatioFormula(r, e.collective)})`]];
    sheet.getRange(`K${r}`).formulas = [[`=J${r}*H${r}`]];
    sheet.getRange(`L${r}`).formulas = [[`=K${r}*G${r}`]];
  }
  const total = first + events.length;
  const totalLabel = model.profile === 'deepseek_v41' ? '所列事件平均 Rank 合计' : '单 Rank 事件合计';
  sheet.getRange(`A${total}:L${total}`).values = [[totalLabel, null, null, null, null, null, null, null, null, null, null, null]];
  sheet.getRange(`K${total}`).formulas = [[`=SUM(K${first}:K${total - 1})`]];
  // Group sizes/membership differ across rows. Their sum is not a group or link total.
  sheet.getRange(`A${total}:L${total}`).format.font = { bold: true };
  sheet.getRange(`E${first}:H${total - 1}`).setNumberFormat("#,##0.####");
  sheet.getRange(`I${first}:L${total}`).setNumberFormat("0.000000");
  styleHeader(sheet, startRow, "L");
  sheet.getRange(`A${startRow}:L${startRow}`).format.rowHeight = 42;
  return { first, last: total, events };
}

function buildPhaseSheet(sheet, model, phase, spec) {
  buildScenario(sheet, model, phase);
  const weights = buildWeightTable(sheet, spec.rows, phase, 1);
  const communication = buildCommunicationTable(sheet, model, phase, spec.facts, weights.last + 2);
  const used = sheet.getUsedRange();
  used.format.font = { name: "Arial", size: 10, color: "#000000" };
  used.format.verticalAlignment = "center";
  setWidths(sheet, [["A:A", 22], ["B:B", 34], ["C:C", 24], ["D:D", 25], ["E:I", 17], ["J:L", 20], ["M:N", 3], ["O:O", 34], ["P:P", 18]]);
  if (model.profile === 'deepseek_v41') {
    sheet.getRange(`B2:B${weights.last}`).format = { wrapText: true, rowHeight: 30, columnWidth: 52 };
    sheet.getRange('O:O').format.columnWidth = 44;
  }
  sheet.freezePanes.freezeRows(1);
  return { weights, communication };
}

function modelReadme(model, weightRows) {
  const slug = modelSlugs[model.key];
  if (model.profile === 'deepseek_v41') return `# ${model.key}\n\nWorkbook: ${slug}-analysis.xlsx\n\nOfficial config revision: ${model.revision}\n\nVA revision: ${model.runtimeRevision}\n\n${weightRows} config-derived weight rows, independently checked against all published tensor headers by npm test. Weights use source storage dtype; this is not VA device residency or proof that the checkpoint loads unchanged.\n\nEditable Prefill/Decode communication covers target attention, uniform hidden-only MoE routing and node-local Engram; other events are explicitly excluded in the workbook.\n\nSee https://github.com/maoxx241/se-tools/blob/main/analysis/DEEPSEEK-V4.1.md for the formulas, versions and runtime limits.\n`;
  return `# ${model.key}\n\n- Workbook: \`${slug}-analysis.xlsx\`\n- Model configuration: \`config.json\`\n- Pinned Hugging Face revision: \`${model.revision}\`\n- Weight rows generated from config and pinned model code: ${weightRows}\n- Evidence and formulas: \`https://github.com/maoxx241/se-tools/blob/main/analysis/EVIDENCE.md\` and \`https://github.com/maoxx241/se-tools/blob/main/analysis/METHODOLOGY.md\`\n\nThe workbook does not read or validate Safetensors metadata. Prefill and Decode each contain editable weight precision, parallel-strategy, request, and communication inputs.\n`;
}

export async function buildModelWorkbook(model, outputRoot, renderRoot = null) {
  const slug = modelSlugs[model.key];
  const modelDir = path.join(outputRoot, slug);
  await fs.mkdir(modelDir, { recursive: true });
  const config = await loadPinnedConfig(model, modelDir);
  const spec = modelSpec(model, config);
  const workbook = Workbook.create();
  const prefill = buildPhaseSheet(workbook.worksheets.add("Prefill"), model, "Prefill", spec);
  const decode = buildPhaseSheet(workbook.worksheets.add("Decode"), model, "Decode", spec);
  if (model.profile === 'deepseek_v41') {
    for (const phase of ['Prefill', 'Decode']) v41WorkbookNotes(workbook.worksheets.getItem(phase), key => sc(phase, key), config);
  }
  workbook.recalculate();
  const outputPath = path.join(modelDir, `${slug}-analysis.xlsx`);
  await (await SpreadsheetFile.exportXlsx(workbook)).save(outputPath);
  await fs.writeFile(path.join(modelDir, "README.md"), modelReadme(model, spec.rows.length), "utf8");
  if (renderRoot) {
    const renderDir = path.join(renderRoot, slug);
    await fs.mkdir(renderDir, { recursive: true });
    for (const name of ["Prefill", "Decode"]) {
      const preview = await workbook.render({ sheetName: name, autoCrop: "all", scale: 0.8, format: "png" });
      await fs.writeFile(path.join(renderDir, `${name}.png`), new Uint8Array(await preview.arrayBuffer()));
    }
  }
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 200 }, maxChars: 12000 });
  await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });
  return { model: model.key, outputPath, weightRows: spec.rows.length, communicationRows: { Prefill: prefill.communication.events.length, Decode: decode.communication.events.length }, errors: errors.ndjson };
}

export async function main() {
  const outputRoot = process.argv[2] || path.resolve("outputs/models");
  const renderRoot = process.argv[3] || null;
  const selected = process.env.MODEL ? models.filter((model) => model.key === process.env.MODEL || modelSlugs[model.key] === process.env.MODEL) : models;
  if (!selected.length) throw new Error(`Unknown MODEL: ${process.env.MODEL}`);
  const results = [];
  for (const model of selected) results.push(await buildModelWorkbook(model, outputRoot, renderRoot));
  console.log(JSON.stringify(results, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
