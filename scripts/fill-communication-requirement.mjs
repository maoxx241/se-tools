import fs from "node:fs/promises";
import path from "node:path";
import { loadCommunicationRequirements } from "../lib/communication-requirements.mjs";
import { loadSpreadsheetRuntime } from "../lib/spreadsheet-runtime.mjs";

const { FileBlob, SpreadsheetFile } = await loadSpreadsheetRuntime();

function formatReadable(bytes) {
  const value = Number(bytes);
  if (bytes >= 1_073_741_824n) return `${(value / 2 ** 30).toFixed(3)} GiB`;
  if (bytes >= 1_048_576n) return `${(value / 2 ** 20).toFixed(3)} MiB`;
  if (bytes >= 1_024n) return `${(value / 2 ** 10).toFixed(3)} KiB`;
  return `${bytes} B`;
}

function byteText(bytes) {
  if (bytes === 0n) return "0 B";
  return `${bytes.toLocaleString("en-US")} B (${formatReadable(bytes)})`;
}

function phaseText(prefill, decode, group) {
  return `P: ${byteText(prefill)}/Step\nD: ${byteText(decode)}/Step\n${group}`;
}

function zeroPhase(group) {
  return `P: 0 B/Step\nD: 0 B/Step\n${group}`;
}

function pdText(cache) {
  return [
    `P→D组: ${byteText(cache.total)}/次`,
    `Attention KV: ${byteText(cache.attention)}`,
    `状态/索引缓存: ${byteText(cache.auxiliary)}`,
  ].join("\n");
}

function moeText(prefill, decode) {
  if (prefill.epAllToAll === 0n) return zeroPhase("无 MoE");
  const lines = [
    `EP32 P: ${byteText(prefill.epAllToAll)}/Step`,
    `EP32 D: ${byteText(decode.epAllToAll)}/Step`,
  ];
  if (prefill.sharedExpertTp > 0n) {
    lines.push(
      `共享专家TP8 P: ${byteText(prefill.sharedExpertTp)}/Step`,
      `共享专家TP8 D: ${byteText(decode.sharedExpertTp)}/Step`,
    );
  } else {
    lines.push("共享专家DP: 0 B/Step（不切TP）");
  }
  lines.push("EP按均匀路由，31/32跨Rank");
  return lines.join("\n");
}

async function loadJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function main() {
  const [source, output, ds10tPath, qwen27Path] = process.argv.slice(2);
  if (!source || !output) {
    throw new Error("Usage: fill-communication-requirement.mjs SOURCE.xlsx OUTPUT.xlsx [DS10T.json QWEN27.json]");
  }
  const overrides = {};
  if (ds10tPath) overrides["deepseek-v4-10t"] = await loadJson(ds10tPath);
  if (qwen27Path) overrides["qwen3.8-27b"] = await loadJson(qwen27Path);
  const results = await loadCommunicationRequirements(process.env.MODEL_CONFIG_ROOT, overrides);

  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
  const audit = {};

  for (const { model: sheetName, facts, dsaModel, prefill, decode, pdCache } of results) {
    const pureDpEpDecode = dsaModel;
    const values = [
      [pureDpEpDecode
        ? "P: 0 B/Step（DP4）\nD: 0 B/Step（DP32）\n推理无跨DP同步"
        : zeroPhase("DP4；推理无跨DP同步")],
      [pureDpEpDecode
        ? `P: ${byteText(prefill.tpBoundary)}/Step（TP8组）\nD: 0 B/Step（TP1）`
        : phaseText(prefill.tpBoundary, decode.tpBoundary, "TP8通信组")],
      [zeroPhase("PP=1")],
      [pureDpEpDecode
        ? `P: ${byteText(prefill.sp)}/Step（SP/TP8组）\nD: 0 B/Step（SP关闭）`
        : phaseText(prefill.sp, decode.sp, "SP启用；TP8通信组")],
      [dsaModel
        ? `P: ${byteText(prefill.dsaCp)}/Step（DSA CP8组）\nD: 0 B/Step（DSA CP关闭）`
        : zeroPhase("PCP=1；未启用")],
      [zeroPhase(dsaModel ? "P的DSA CP已计入上一行；DCP关闭" : "DCP=1；未启用")],
      [`P: ${byteText(prefill.attention)}/Step\n${dsaModel ? "DSA CP8通信组" : "SP/TP8通信组"}`],
      [`D: ${byteText(decode.attention)}/Step\n${pureDpEpDecode ? "TP1；无Attention集合通信" : "SP/TP8通信组"}`],
      [pdText(pdCache)],
      ["0 B；仅统计P→D"],
      ["0 B；仅统计P→D"],
      ["0 B；仅统计P→D"],
      [moeText(prefill, decode)],
      ["0 B/Step；内置投机无独立P2P\n7步已计入D: 128 token/Step"],
      ["0 B/Step；模型内驻留表，无独立通信"],
    ];

    const sheet = workbook.worksheets.getItem(sheetName);
    sheet.getRange("I4:I18").values = values;
    sheet.getRange("I4:I18").format.wrapText = true;
    sheet.getRange("I4:I18").format.verticalAlignment = "center";
    sheet.getRange("I4:I18").format.horizontalAlignment = "left";
    sheet.getRange("I4:I18").format.font = { name: "Arial", size: 8, color: "#000000" };
    sheet.getRange("I:I").format.columnWidth = 33;
    sheet.getRange("4:18").format.rowHeight = 56;
    sheet.getRange("12:12").format.rowHeight = 72;
    sheet.getRange("16:16").format.rowHeight = 94;

    audit[sheetName] = {
      facts,
      prefill: Object.fromEntries(Object.entries(prefill).map(([key, value]) => [key, value.toString()])),
      decode: Object.fromEntries(Object.entries(decode).map(([key, value]) => [key, value.toString()])),
      pdCache: Object.fromEntries(Object.entries(pdCache).map(([key, value]) => [key, value.toString()])),
    };
  }

  await fs.mkdir(path.dirname(output), { recursive: true });
  workbook.recalculate();
  await (await SpreadsheetFile.exportXlsx(workbook)).save(output);
  console.log(JSON.stringify({ output, audit }, null, 2));
}

await main();
