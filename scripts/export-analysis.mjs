import fs from "node:fs/promises";
import { models, modelSlugs, readModelConfig } from "../lib/model-catalog.mjs";
import { modelSpec, elementCount } from "./model-analysis-specs.mjs";
import { loadCommunicationRequirements } from "../lib/communication-requirements.mjs";
import { estimateCollective, estimateAllToAllV, jsonBytes } from "../lib/collectives.mjs";
import { v41Step, v41Cache, v41Topology, v41EngramRouting } from '../lib/deepseek-v41.mjs';
import { loadKvEpSpecs } from '../lib/kv-ep-specs.mjs';

const [command, input] = process.argv.slice(2);
if (!command || ["help", "--help", "-h"].includes(command)) {
  console.log(`Usage:
  node scripts/export-analysis.mjs weights [MODEL_SLUG]
  node scripts/export-analysis.mjs communication [deepseek-v4.1-flash]
  node scripts/export-analysis.mjs kv-ep
  node scripts/export-analysis.mjs engram EVENT.json
  node scripts/export-analysis.mjs collective EVENT.json

Uses checked-in configs; no network, NPU, or spreadsheet package required.
Large byte counts are serialized as decimal strings. Excel inputs remain in Excel.
MODEL_CONFIG_ROOT optionally selects a directory containing <slug>/config.json.`);
} else if (command === "weights") {
  const selected = input ? models.filter((m) => m.key === input || modelSlugs[m.key] === input) : models;
  if (!selected.length) throw new Error(`Unknown weight model: ${input}`);
  const results = [];
  for (const model of selected) {
    const slug = modelSlugs[model.key];
    const spec = modelSpec(model, await readModelConfig(slug));
    const rows = spec.rows.map((row) => {
      const elements = elementCount(row.dims) * row.count;
      const bytes = elements * row.bytes;
      if (!Number.isSafeInteger(elements) || !Number.isSafeInteger(bytes)) throw new Error(`Invalid tensor dimensions/bytes: ${row.name}`);
      return { ...row, totalElements: elements, totalBytes: BigInt(bytes) };
    });
    results.push({ model: model.key, slug, facts: spec.facts, rows,
      totalWeightBytes: rows.reduce((s, row) => s + row.totalBytes, 0n),
      scope: "config/code weight model; rank partitioning is editable in Excel; not checkpoint or device residency bytes" });
  }
  console.log(jsonBytes(results));
} else if (command === "communication") {
  if (input === 'deepseek-v4.1-flash') {
    const config = await readModelConfig(input);
    console.log(jsonBytes({ model: 'DeepSeek-V4.1-Flash', topology: v41Topology(config),
      Prefill: v41Step(config, { tokens: 16384 }),
      Decode: v41Step(config, { tokens: 128, tp: 1, dp: 32, ep: 32, sp: false, dsaCP: false }),
      cache: v41Cache(config),
    }));
  } else if (input) throw new Error(`Unknown communication example: ${input}`);
  else console.log(jsonBytes(await loadCommunicationRequirements()));
} else if (command === 'kv-ep') {
  console.log(jsonBytes(await loadKvEpSpecs()));
} else if (command === 'engram') {
  if (!input) throw new Error('engram requires EVENT.json');
  console.log(jsonBytes(v41EngramRouting(JSON.parse(await fs.readFile(input, 'utf8')))));
} else if (command === "collective") {
  if (!input) throw new Error("collective requires EVENT.json");
  const event = JSON.parse(await fs.readFile(input, "utf8"));
  console.log(jsonBytes(event.collective === "AlltoAllV" ? estimateAllToAllV(event) : estimateCollective(event)));
} else throw new Error(`Unknown command: ${command}; use --help`);
