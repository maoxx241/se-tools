import assert from 'node:assert/strict';
import { cacheBytes, evaluateMemory, GiB } from '../lib/model-memory/engine.mjs';
import { inputRows } from './memory-workbook.mjs';

const scalar = (s, address) => s.getRange(address).values[0][0];
function close(actual, expected, label) {
  assert.ok(typeof actual === 'number' && Number.isFinite(actual)
    && Math.abs(actual - expected) <= 1e-10 * Math.max(1, Math.abs(expected)), `${label}: ${actual} != ${expected}`);
}

export function verifyMemorySheet(model, item) {
  const { sheet: s, preset, rows, maxRow, groupTotal, stageCols, helperCols } = item;
  const inputs = Object.fromEntries(Object.entries(inputRows).map(([key, n]) => [key, scalar(s, `R${n}`)]));
  const stage = scalar(s, 'R22'), data = s.getRange(`A2:${helperCols.at(-1)}${item.end}`).values;
  const overrides = Object.fromEntries(rows.map((r, i) => [`${r.group}/${r.name}`, data[i][3]]));
  const expected = evaluateMemory(model.spec, rows, preset, inputs, overrides), active = expected.stages[stage];
  assert.ok(active, `Invalid PP stage: ${stage}`);
  assert.equal(scalar(s, 'R30'), 1, `${s.name} input validity`);
  for (const [i, r] of active.rows.entries()) {
    const n = i + 2, actual = data[i];
    assert.equal(actual[0], rows[i].group);
    assert.equal(actual[1], rows[i].name);
    assert.deepEqual(JSON.parse(actual[2]), rows[i].shape);
    assert.equal(actual[4], rows[i].repeat);
    close(actual[9], r.globalBytes / GiB, `${s.name} J${n} global`);
    close(actual[10], r.bytes / GiB, `${s.name} K${n} local`);
    close(actual[12], r.divisor, `${s.name} M${n} divisor`);
    assert.deepEqual(JSON.parse(actual[13]), r.shape);
    assert.equal(actual[14], r.repeat);
    for (const [j, st] of expected.stages.entries()) {
      close(scalar(s, `${stageCols[j]}${n}`), st.rows[i].bytes / GiB, `${s.name} PP${j}/${r.name}`);
    }
    for (const [j, key] of ['weight', 'scale', 'auxiliary'].entries()) {
      close(scalar(s, `${helperCols[j]}${n}`), r[key], `${s.name} ${key}/${r.name}`);
    }
    assert.ok(s.getRange(`K${n}`).formulas[0][0], `${s.name} K${n} must remain a formula`);
  }
  for (const [address, bytes] of [['R23', active.weight], ['R25', active.buffer], ['R27', active.cache.total],
    ['R28', active.total], [`V${groupTotal}`, active.weight], [`W${maxRow}`, expected.maxWeight],
    [`X${maxRow}`, expected.maxCache], [`Y${maxRow}`, expected.maxTotal]]) {
    close(scalar(s, address), bytes / GiB, `${s.name} ${address}`);
  }
  close(scalar(s, 'R24'), (active.weight + active.cache.total) / GiB, `${s.name} weight+cache`);
  for (const [i, st] of expected.stages.entries()) {
    for (const [col, bytes] of [['W', st.weight], ['X', st.cache.total], ['Y', st.total], ['AC', st.buffer]]) {
      close(scalar(s, `${col}${item.stageFirstRow + i}`), bytes / GiB, `${s.name} ${col} PP${i}`);
    }
  }
  for (let n = 63; n <= 72; n++) {
    const result = cacheBytes(model.spec.facts, { ...inputs, tokens: scalar(s, `Q${n}`) }, active);
    const actual = s.getRange(`R${n}:Y${n}`).values[0];
    close(actual[0], result.localTokens, `${s.name} cache tokens/${n}`);
    for (const [j, key] of ['latent', 'rope', 'indexer', 'recurrent', 'conv', 'total'].entries()) {
      close(actual[j + 1], result[key] / GiB, `${s.name} cache ${key}/${n}`);
    }
    close(actual[7], (active.weight + result.total) / GiB, `${s.name} cache/weight/${n}`);
  }
  const structures = s.getRange(`R40:V${groupTotal - 1}`).values;
  for (const r of structures) close(r[0] * r[1], r[2], `${s.name} structure repetitions`);
  close(structures.reduce((total, r) => total + r[4], 0), active.weight / GiB, `${s.name} structure sum`);
  assert.equal(scalar(s, 'T15'), inputs.blockSize);
  close(scalar(s, 'T24'), inputs.ep / inputs.tp, `${s.name} DP`);
  assert.equal(scalar(s, 'T27'), inputs.ep * expected.stages.length);
  assert.ok(!s.getRange('R10').formulas[0][0], `${s.name} shared TP default must be independent`);
  return { sheet: s.name, stageCount: expected.stages.length, weightGiB: active.weight / GiB,
    cacheGiB: active.cache.total / GiB, bufferGiB: active.buffer / GiB, totalGiB: active.total / GiB,
    maxTotalGiB: expected.maxTotal / GiB };
}

export function sweepMemoryInputs(model, item, workbook) {
  const { sheet: s, rows, preset } = item;
  let checks = 0;
  const change = (address, next) => {
    const range = s.getRange(address), saved = { value: range.values[0][0], formula: range.formulas[0][0] };
    range.values = [[next]];
    try { workbook.recalculate(); verifyMemorySheet(model, item); checks++; }
    finally {
      if (saved.formula) range.formulas = [[saved.formula]];
      else range.values = [[saved.value]];
      workbook.recalculate();
    }
  };
  for (const [i] of rows.entries()) change(`D${i + 2}`, scalar(s, `D${i + 2}`) === 'BF16' ? 'MXFP4' : 'BF16');
  const shared = scalar(s, 'R10');
  change('R2', scalar(s, 'R2') / 2);
  assert.equal(scalar(s, 'R10'), shared);
  for (const n of [3, 4, 5, 6, 7, 8, 10]) change(`R${n}`, scalar(s, `R${n}`) > 1 ? scalar(s, `R${n}`) / 2 : 2);
  for (const ep of [scalar(s, 'R2'), 128, 160, 1024]) change('R9', ep);
  if (preset.phase === 'Prefill' && model.spec.facts.hasDSA) change('R11', 1);
  if (preset.phase === 'Decode' && scalar(s, 'R12') > 1) change('R12', scalar(s, 'R12') / 2);
  if (model.spec.facts.hasDSA) {
    const old = s.getRange('R13:R14').values;
    try {
      for (const mla of ['BF16', 'INT8', 'FP8']) for (const index of ['BF16', 'INT8', 'FP8']) {
        s.getRange('R13:R14').values = [[mla], [index]];
        workbook.recalculate(); verifyMemorySheet(model, item); checks++;
      }
    } finally { s.getRange('R13:R14').values = old; workbook.recalculate(); }
  }
  const boundary = scalar(s, 'R19') * scalar(s, 'R12');
  for (const tokens of new Set([1, Math.max(1, boundary - 1), boundary, boundary + 1, 262145, model.spec.facts.maxTokens]
    .filter(n => n <= model.spec.facts.maxTokens))) change('R18', tokens);
  change('R19', scalar(s, 'R19') * 2); change('R20', 3);
  for (let n = 0; n < item.ownership.length; n++) change('R22', n);
  for (const [address, invalid] of [['R2', 7], ['R3', 7], ['R18', 0], ['R19', 0], ['R22', item.ownership.length]]) {
    const range = s.getRange(address), saved = { value: range.values[0][0], formula: range.formulas[0][0] };
    range.values = [[invalid]]; workbook.recalculate();
    try {
      assert.equal(scalar(s, 'R30'), 0);
      assert.equal(typeof scalar(s, 'R28'), 'string', `${s.name} invalid inputs must not produce a numeric total`);
      checks++;
    } finally {
      if (saved.formula) range.formulas = [[saved.formula]];
      else range.values = [[saved.value]];
      workbook.recalculate();
    }
  }
  verifyMemorySheet(model, item);
  return checks;
}
