import assert from 'node:assert/strict';
import test from 'node:test';
import { loadMemoryModel, loadMemoryDescription } from '../lib/model-memory/catalog.mjs';
import { kimiMemorySpec } from '../lib/model-memory/kimi.mjs';
import { selectMemoryRows, validateMemorySpec } from '../lib/model-memory/schema.mjs';
import { cacheBytes, evaluateMemory, GiB, resolveInputs, stageOwnership, storageBytes } from '../lib/model-memory/engine.mjs';
import { parseMemoryArgs, memoryReport } from '../scripts/build-memory-workbook.mjs';

const publicModel = await loadMemoryModel('kimi-k3');
const synthetic = await loadMemoryDescription(new URL('../examples/model-memory/synthetic.json', import.meta.url));
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-10 * Math.max(1, Math.abs(b)), `${a} != ${b}`);

test('public archived model generates four scopes/phases without inheriting DSA/indexer rules', () => {
  assert.equal(publicModel.scenarios.length, 4);
  const full = publicModel.scenarios.find(s => s.preset.name === '全量含ViT_Decode');
  const result = evaluateMemory(publicModel.spec, full.rows, full.preset);
  assert.equal(publicModel.spec.facts.hasDSA, false);
  assert.equal(result.stages[0].cache.indexer, 0);
  near(result.stages[0].cache.total / GiB, 3.4011569023132324);
  near(result.maxTotal / GiB, 36.81021534651518);
  assert.equal(full.rows.filter(r => r.group === 'DSA Indexer').length, 0);
  const prefill = publicModel.scenarios.find(s => s.preset.name === '全量含ViT_Prefill');
  assert.throws(() => evaluateMemory(publicModel.spec, prefill.rows, prefill.preset, { dsaCP: 16 }), /DSA CP/);
  assert.throws(() => evaluateMemory(publicModel.spec, full.rows, full.preset, { mlaDtype: 'INT8' }), /BF16 only/);
});

test('invented six-layer model agrees with hand-counted tensor, cache, and gather bytes', () => {
  const { preset, rows } = synthetic.scenarios[0], result = evaluateMemory(synthetic.spec, rows, preset).stages[0];
  const row = name => result.rows.find(r => r.name.endsWith(`/${name}`));
  assert.equal(row('embed_tokens.weight').bytes, 131072);
  assert.equal(row('input_layernorm.weight').bytes, 3072);
  assert.equal(row('experts.*.w1').bytes, 5 * (64 * 128 / 2 + 64 * 4));
  assert.equal(row('shared_experts.down_proj').bytes, 5 * 256 * 16 * 2);
  assert.equal(result.weight, 652032);
  assert.equal(result.cache.total, 87680);
  assert.equal(result.buffer, 98304);
  assert.equal(result.total, 838016);
});

test('MXFP scales and packed INT4 use local geometry, including partial groups and per-channel metadata', () => {
  assert.deepEqual(storageBytes([2, 33], 'MXFP4'), { weight: 34, scale: 4, auxiliary: 0, total: 38 });
  assert.deepEqual(storageBytes([2, 33], 'MXFP8'), { weight: 66, scale: 4, auxiliary: 0, total: 70 });
  assert.deepEqual(storageBytes([2, 33], 'INT4'), { weight: 34, scale: 0, auxiliary: 32, total: 66 });
  assert.equal(storageBytes([2, 32], 'FP8').scale, 0);
  assert.throws(() => storageBytes([1.5, 16], 'BF16'), /Invalid local/);
  assert.throws(() => storageBytes([2, 32], 'unknown'), /Unsupported dtype/);
  const item = synthetic.scenarios[2], result = evaluateMemory(synthetic.spec, item.rows, item.preset);
  const shared = result.stages[1].rows.find(r => r.name === 'MoE/shared_experts.down_proj');
  assert.deepEqual(shared.shape, [256, 16]);
  assert.equal(shared.scale, 256);
  assert.equal(shared.weight, 256 * 8);
});

test('DSA CP does not reduce retained prefill cache; DCP shards MLA and keeps replicated indexer', () => {
  const prefill = synthetic.scenarios[0], decode = synthetic.scenarios[1];
  const p = evaluateMemory(synthetic.spec, prefill.rows, prefill.preset);
  const noCP = evaluateMemory(synthetic.spec, prefill.rows, prefill.preset, { dsaCP: 1 });
  assert.equal(p.stages[0].cache.total, noCP.stages[0].cache.total);
  const d = evaluateMemory(synthetic.spec, decode.rows, decode.preset, { tokens: 256 });
  const noDCP = evaluateMemory(synthetic.spec, decode.rows, decode.preset, { tokens: 256, dcp: 1 });
  assert.equal(d.stages[0].cache.indexer, noDCP.stages[0].cache.indexer);
  assert.equal(d.stages[0].cache.latent * 4, noDCP.stages[0].cache.latent);
  assert.equal(d.stages[0].cache.recurrent, noDCP.stages[0].cache.recurrent);
});

test('all nine cache dtype combinations, scales, block boundaries, and endpoint states are independent', () => {
  const item = synthetic.scenarios[1], p = resolveInputs(item.preset), owner = stageOwnership(synthetic.spec.facts, [6])[0];
  for (const mlaDtype of ['BF16', 'INT8', 'FP8']) for (const indexDtype of ['BF16', 'INT8', 'FP8']) {
    const result = cacheBytes(synthetic.spec.facts, { ...p, mlaDtype, indexDtype }, owner);
    assert.equal(result.latentPerToken + result.ropePerToken, mlaDtype === 'BF16' ? 160 : 100);
    assert.equal(result.indexPerToken, indexDtype === 'BF16' ? 64 : indexDtype === 'INT8' ? 34 : 36);
  }
  const boundary = p.blockSize * p.dcp;
  const at = cacheBytes(synthetic.spec.facts, { ...p, tokens: boundary }, owner);
  const above = cacheBytes(synthetic.spec.facts, { ...p, tokens: boundary + 1 }, owner);
  assert.equal(above.localTokens, at.localTokens * 2);
  assert.equal(above.recurrent, at.recurrent); assert.equal(above.conv, at.conv);
});

test('shared TP defaults remain independent; EP accounts for uneven and excess expert slots', () => {
  const { preset, rows } = synthetic.scenarios[1];
  assert.equal(resolveInputs(preset, { tp: 2 }).sharedTP, 4);
  const ep8 = evaluateMemory(synthetic.spec, rows, preset, { ep: 8 });
  const ep12 = evaluateMemory(synthetic.spec, rows, preset, { ep: 12 });
  const expert = rows.findIndex(r => r.rule === 'EP');
  assert.equal(ep8.stages[0].rows[expert].repeat, 5 * 2);
  assert.equal(ep12.stages[0].rows[expert].repeat, 5);
  const shared = rows.findIndex(r => r.shared), unsplit = evaluateMemory(synthetic.spec, rows, preset, { sharedTP: 1 });
  assert.equal(unsplit.stages[0].rows[shared].bytes, ep12.stages[0].rows[shared].bytes * 4);
});

test('uneven PP stages, including an attention-free stage, own only their layers and endpoint tensors', () => {
  const { preset, rows } = synthetic.scenarios[2];
  const result = evaluateMemory(synthetic.spec, rows, preset);
  assert.deepEqual(result.stages.map(s => s.attention), [0, 2]);
  assert.deepEqual(result.stages.map(s => s.kda), [2, 2]);
  assert.equal(result.stages[0].buffer, 0);
  assert.ok(result.stages[1].buffer > 0);
  const embedding = rows.findIndex(r => r.name === 'embed_tokens.weight'), head = rows.findIndex(r => r.name === 'lm_head.weight');
  assert.deepEqual(result.stages.map(s => s.rows[embedding].repeat), [1, 0]);
  assert.deepEqual(result.stages.map(s => s.rows[head].repeat), [0, 1]);
  assert.equal(result.maxTotal, Math.max(...result.stages.map(s => s.total)));
  assert.throws(() => stageOwnership(synthetic.spec.facts, [2, 2]), /exactly once/);
});

test('infeasible topology, missing layers, unknown rules and unknown recipes fail explicitly', async () => {
  await assert.rejects(loadMemoryModel('unknown'), /Unsupported memory model/);
  assert.throws(() => selectMemoryRows(synthetic.spec, 'guess'), /Unknown quantization/);
  const { preset, rows } = synthetic.scenarios[0];
  for (const changes of [{ tp: 3 }, { ep: 13 }, { dcp: 2 }, { tokens: 0 }, { tokens: 4097 }, { mlaDtype: 'C8' }]) {
    assert.throws(() => evaluateMemory(synthetic.spec, rows, preset, changes));
  }
  const bad = structuredClone(synthetic.spec);
  bad.rows[0].rule = 'infer';
  assert.throws(() => validateMemorySpec(bad), /Unsupported partition/);
  const broken = structuredClone(publicModel.config);
  broken.text_config.linear_attn_config.kda_layers.pop();
  assert.throws(() => kimiMemorySpec(broken), /partition all/);
});

test('source declarations and recipe-specific formats stay separate from deployment inputs', () => {
  const source = selectMemoryRows(synthetic.spec, 'checkpoint'), quantized = selectMemoryRows(synthetic.spec, 'mxfp');
  const index = source.findIndex(r => r.group === 'KDA' && r.name === 'q_proj');
  assert.equal(source[index].dtype, 'BF16');
  assert.equal(quantized[index].dtype, 'MXFP8');
  assert.deepEqual(source[index].shape, quantized[index].shape);
});

test('numeric reports are deterministic; CLI supports separate descriptions and protects published artifacts', () => {
  assert.deepEqual(memoryReport(publicModel), memoryReport(publicModel));
  assert.equal(parseMemoryArgs(['--model', 'kimi-k3']).models[0], 'kimi-k3');
  assert.ok(parseMemoryArgs(['--spec', 'example.json']).specPath.endsWith('example.json'));
  assert.throws(() => parseMemoryArgs(['--spec', 'example.json', '--model', 'kimi-k3']), /mutually exclusive/);
  assert.throws(() => parseMemoryArgs(['--output', 'models/kimi-k3']), /publication/);
  assert.throws(() => parseMemoryArgs(['--model']), /Missing value/);
});
