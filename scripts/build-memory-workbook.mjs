import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadMemoryModel, loadMemoryDescription, memoryModels } from '../lib/model-memory/catalog.mjs';
import { evaluateMemory, GiB } from '../lib/model-memory/engine.mjs';
import { loadSpreadsheetRuntime } from '../lib/spreadsheet-runtime.mjs';
import { createMemoryWorkbook } from './memory-workbook.mjs';
import { sweepMemoryInputs, verifyMemorySheet } from './verify-memory-sheet.mjs';

export function parseMemoryArgs(args) {
  const options = { output: path.resolve('outputs/memory'), models: memoryModels, render: false, sweep: true };
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--help' || arg === '-h') { options.help = true; return options; }
    if (arg === '--render') options.render = true;
    else if (arg === '--no-sweep') options.sweep = false;
    else if (['--model', '--output', '--presets', '--spec'].includes(arg)) {
      const next = args[++i];
      if (!next || next.startsWith('--')) throw new Error(`Missing value for ${arg}`);
      if (arg === '--model') options.models = next === 'all' ? memoryModels : [next];
      else if (arg === '--output') options.output = path.resolve(next);
      else if (arg === '--spec') options.specPath = path.resolve(next);
      else options.presetsPath = path.resolve(next);
    } else throw new Error(`Unknown argument: ${arg}`);
  }
  if (options.specPath && args.includes('--model')) throw new Error('--spec and --model are mutually exclusive');
  if (options.presetsPath && !options.specPath && options.models.length !== 1) throw new Error('--presets requires one --model');
  const published = fileURLToPath(new URL('../models/', import.meta.url));
  if (options.output === published.slice(0, -1) || options.output.startsWith(published)) throw new Error('Generate into a build directory; publication is a separate explicit file copy');
  return options;
}

export function memoryReport(model) {
  return {
    schemaVersion: 1, model: model.slug, ...model.provenance,
    evidence: model.spec.evidence ?? 'Explicit model description; no implicit source verification',
    scope: 'Weights + one request cache + explicitly modeled o_proj buffers; excludes other activations, graph/workspace and allocator/cache-pool reservation',
    scenarios: model.scenarios.map(({ preset, rows }) => {
      const result = evaluateMemory(model.spec, rows, preset);
      return { name: preset.name, phase: preset.phase, recipe: preset.recipe, scope: preset.scope,
        inputs: result.inputs, partitions: preset.partitions, maxTotalGiB: result.maxTotal / GiB,
        stages: result.stages.map(st => ({ stage: st.stage, layers: [st.start, st.end], kdaLayers: st.kda,
          attentionLayers: st.attention, weightGiB: st.weight / GiB, cacheGiB: st.cache.total / GiB,
          bufferGiB: st.buffer / GiB, totalGiB: st.total / GiB,
          rows: st.rows.map(r => ({ name: r.name, dtype: r.dtype, shape: r.shape, repeat: r.repeat,
            bytes: r.bytes, globalBytes: r.globalBytes })) })) };
    }),
  };
}

export async function buildMemory(options) {
  const models = options.specPath ? [await loadMemoryDescription(options.specPath, options)]
    : await Promise.all(options.models.map(slug => loadMemoryModel(slug, options)));
  const { Workbook, SpreadsheetFile } = await loadSpreadsheetRuntime();
  const result = [];
  for (const model of models) {
    const { workbook, sheets } = createMemoryWorkbook(model, Workbook);
    const verification = sheets.map(item => {
      const summary = verifyMemorySheet(model, item);
      return { ...summary, inputChecks: options.sweep ? sweepMemoryInputs(model, item, workbook) : 0 };
    });
    workbook.recalculate();
    const errors = await workbook.inspect({ kind: 'match',
      searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
      options: { useRegex: true, maxResults: 20 }, maxChars: 4000 });
    if (!errors.ndjson.includes('matched 0 entries')) throw new Error(`Formula errors in ${model.slug}: ${errors.ndjson}`);
    const report = memoryReport(model);
    const directory = path.join(options.output, model.slug);
    await fs.mkdir(directory, { recursive: true });
    const outputPath = path.join(directory, `${model.slug}-memory.xlsx`);
    const temporary = path.join(directory, `${model.slug}-memory.building.xlsx`);
    try {
      await (await SpreadsheetFile.exportXlsx(workbook)).save(temporary);
      // Verify the actual saved formulas and cached results before publishing this build.
      const { verifySavedMemory } = await import('./verify-memory-workbook.mjs');
      await verifySavedMemory(model, temporary, { sweep: false });
      await fs.rename(temporary, outputPath);
    } finally { await fs.rm(temporary, { force: true }); }
    await fs.writeFile(path.join(directory, 'memory-results.json'), `${JSON.stringify(report, null, 2)}\n`);
    await fs.writeFile(path.join(directory, 'memory-verification.json'), `${JSON.stringify({ model: model.slug,
      checks: verification, formulaErrors: 0, savedFormulasAndCaches: 'verified' }, null, 2)}\n`);
    if (options.render) {
      const previewDir = path.join(directory, 'previews');
      await fs.mkdir(previewDir, { recursive: true });
      for (const item of sheets) for (const [part, range] of [
        ['weights', `A1:O${item.end}`], ['settings', 'Q1:V38'], ['structures', `Q39:X${item.groupTotal}`],
        ['cache', 'Q62:Y82'], ['pp', `Q${item.stageHeaderRow}:AC${item.maxRow}`],
      ]) {
        const preview = await workbook.render({ sheetName: item.sheet.name, range, scale: 1, format: 'png' });
        await fs.writeFile(path.join(previewDir, `${item.sheet.name}-${part}.png`), new Uint8Array(await preview.arrayBuffer()));
      }
    }
    result.push({ model: model.slug, sheets: sheets.length, inputChecks: verification.reduce((n, r) => n + r.inputChecks, 0), outputPath });
  }
  return result;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const options = parseMemoryArgs(process.argv.slice(2));
  if (options.help) console.log('Usage: node scripts/build-memory-workbook.mjs [--model kimi-k3|all | --spec MODEL.json] [--output DIR] [--presets FILE.json] [--render] [--no-sweep]\nDefault: public archived models, full input sweep, outputs/memory. Existing model archives are never overwritten.');
  else console.log(JSON.stringify(await buildMemory(options), null, 2));
}
