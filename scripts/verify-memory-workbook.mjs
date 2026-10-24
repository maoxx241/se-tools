import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { loadMemoryModel, loadMemoryDescription } from '../lib/model-memory/catalog.mjs';
import { loadSpreadsheetRuntime } from '../lib/spreadsheet-runtime.mjs';
import { createMemoryWorkbook } from './memory-workbook.mjs';
import { sweepMemoryInputs, verifyMemorySheet } from './verify-memory-sheet.mjs';

const normalize = formula => formula?.replace(/^=/, '').replaceAll('_xlfn.', '');
const close = (a, b, label) => assert.ok(typeof a === 'number' && Math.abs(a - b) < 1e-9 * Math.max(1, Math.abs(b)), `${label}: ${a} != ${b}`);

export async function verifySavedMemory(model, file, { sweep = true } = {}) {
  const native = JSON.parse(execFileSync(process.env.PYTHON || 'python3',
    [fileURLToPath(new URL('./read-memory-xlsx.py', import.meta.url)), file], { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 }));
  const { Workbook, SpreadsheetFile, FileBlob } = await loadSpreadsheetRuntime();
  const expected = createMemoryWorkbook(model, Workbook);
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
  assert.deepEqual(native.sheets.map(s => s.name), expected.sheets.map(s => s.sheet.name), 'Saved sheet order/coverage');
  const summaries = [];
  for (const [index, item] of expected.sheets.entries()) {
    const actual = native.sheets[index], sheet = workbook.worksheets.getItem(actual.name);
    const expectedRange = item.sheet.getRange(`A1:${item.helperCols.at(-1)}${Math.max(98, item.maxRow)}`);
    const formulas = expectedRange.formulas, values = expectedRange.values;
    let formulaCount = 0;
    for (let row = 0; row < formulas.length; row++) for (let col = 0; col < formulas[row].length; col++) {
      let letters = '', n = col + 1;
      while (n) { letters = String.fromCharCode(65 + (n - 1) % 26) + letters; n = Math.floor((n - 1) / 26); }
      const address = `${letters}${row + 1}`, cell = actual.cells[address];
      const value = values[row][col], formula = formulas[row][col];
      if (formula) {
        assert.ok(cell?.formula, `${actual.name}!${address} formula lost on export`);
        assert.equal(normalize(cell.formula), normalize(formula), `${actual.name}!${address} formula changed on export`);
        formulaCount++;
      }
      if (typeof value === 'number') close(cell?.value, value, `${actual.name}!${address} cached value`);
      else if (value != null && value !== '') assert.equal(cell?.value, value, `${actual.name}!${address} saved value`);
    }
    assert.ok(actual.merges.includes('U1:V2'), `${actual.name} signature merge`);
    for (const range of ['R13', 'R14', 'R18', 'R22', `D2:D${item.end}`]) {
      assert.ok(actual.validations.includes(range), `${actual.name} validation ${range}`);
    }
    assert.equal(actual.pane?.xSplit, '2', `${actual.name} frozen identifier columns`);
    assert.equal(actual.pane?.ySplit, '1', `${actual.name} frozen header row`);
    // Restore only formulas independently found and checked in the saved OOXML.
    // This avoids the importer treating shared-formula children as constants.
    for (const [address, cell] of Object.entries(actual.cells)) if (cell.formula) sheet.getRange(address).formulas = [[cell.formula]];
    workbook.recalculate();
    const imported = { ...item, sheet };
    const summary = verifyMemorySheet(model, imported);
    summaries.push({ ...summary, savedFormulas: formulaCount, inputChecks: sweep ? sweepMemoryInputs(model, imported, workbook) : 0 });
  }
  workbook.recalculate();
  const errors = await workbook.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
    options: { useRegex: true, maxResults: 20 }, maxChars: 4000 });
  assert.match(errors.ndjson, /matched 0 entries/);
  await fs.rm(`${file}.inspect.ndjson`, { force: true });
  return { model: model.slug, savedFormulasAndCaches: 'verified', formulaErrors: 0, sheets: summaries };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), presetIndex = args.indexOf('--presets');
  let presetsPath;
  if (presetIndex >= 0) {
    if (!args[presetIndex + 1] || args[presetIndex + 1].startsWith('--')) throw new Error('--presets requires a JSON file');
    presetsPath = path.resolve(args[presetIndex + 1]);
    args.splice(presetIndex, 2);
  }
  const [slug, source, third] = args;
  const file = slug === '--spec' ? third : source;
  if (!slug || !file) throw new Error('Usage: node scripts/verify-memory-workbook.mjs MODEL_SLUG FILE.xlsx [--no-sweep], or --spec MODEL.json FILE.xlsx');
  const model = slug === '--spec' ? await loadMemoryDescription(source, { presetsPath }) : await loadMemoryModel(slug, { presetsPath });
  const result = await verifySavedMemory(model, path.resolve(file), { sweep: !process.argv.includes('--no-sweep') });
  console.log(JSON.stringify(result, null, 2));
}
