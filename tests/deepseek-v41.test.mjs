import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { v41Topology, v41WeightSpec, v41Cache, v41Step, v41EngramRouting } from '../lib/deepseek-v41.mjs';
const root = new URL('../models/deepseek-v4.1-flash/', import.meta.url);
const configBytes = await fs.readFile(new URL('config.json', root));
const config = JSON.parse(configBytes);
const metadata = JSON.parse(await fs.readFile(new URL('weight-metadata.json', root), 'utf8'));

test('V4.1 generated shapes/dtypes/counts match all 96,085 published tensor headers', () => {
  assert.equal(createHash('sha256').update(configBytes).digest('hex'), metadata.configSha256);
  const dtype = { BF16: 'BF16', F32: 'F32', 'FP8 E4M3': 'F8_E4M3', E8M0: 'F8_E8M0', 'FP4 E2M1': 'I8' };
  const actual = v41WeightSpec(config).rows.map(r => ({
    name: r.name.replace(/\.\d+\./g, '.N.'), dtype: dtype[r.dtype],
    storedShape: r.dtype === 'FP4 E2M1' ? [r.dims[0], r.dims[1] / 2] : r.dims,
    bytesPerTensor: r.dims.reduce((p, x) => p * x, r.bytes), count: r.count,
  })).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
  const expected = metadata.groups.toSorted((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
  assert.deepEqual(actual, expected);
  assert.equal(actual.reduce((sum, r) => sum + r.bytesPerTensor * r.count, 0), 510286023000);
  const changed = structuredClone(config); changed.text_config.hidden_size = 10240;
  assert.equal(v41WeightSpec(changed).rows.find(r => r.name === 'embed.weight').dims[1], 10240);
});

test('V4.1 source sharing and BF16 retained cache exclude duplicate consumer planes', () => {
  const topology = v41Topology(config);
  assert.deepEqual(topology.kvSources, [2, 8, 14, 20]);
  assert.equal(topology.layers[26].kvSource, 20);
  assert.equal(topology.layers[26].indexSource, 24);
  assert.equal(topology.layers[26].candidateFilter, true);
  const cache = v41Cache(config, { sequenceTokens: 262144, requests: 16, tp: 8 });
  assert.equal(cache.longRowsPerRequest, 655360n);
  assert.equal(cache.longKVBytesPerRequest, 671088640n);
  assert.equal(cache.indexBytesPerRequest, 85196800n);
  assert.equal(cache.ringBytesPerRequest, 393216n);
  assert.equal(cache.batchLogicalBytesAcrossTP, cache.batchLogicalBytesPerReplica * 8n);
  assert.equal(cache.pdTransferBytes, null);
  assert.equal(v41Cache(config, { sequenceTokens: 3 }).longRowsPerRequest, 6n);
  const broken = structuredClone(config); broken.text_config.compress_ratios[8] = 1;
  assert.throws(() => v41Topology(broken), /KV source/);
});

test('V4.1 CP head exchange is additional to SP, without TopK or shared-expert gathers', () => {
  const cp = v41Step(config, { tokens: 16384 });
  const tp = v41Step(config, { tokens: 16384, dsaCP: false });
  assert.equal(cp.attention.spInputGroupBytes, 46976204800n);
  assert.equal(cp.attention.headExchangeGroupBytes, 37580963840n);
  assert.equal(tp.attention.headExchangeGroupBytes, 0n);
  assert.equal(cp.attention.spInputGroupBytes, tp.attention.spInputGroupBytes);
  assert.equal(cp.attention.indexTopKGroupBytes, 0n);
  assert.equal(cp.attention.sharedExpertGroupBytes, 0n);
  const single = v41Step(config, { tokens: 128, tp: 1, dp: 32, sp: false, dsaCP: false });
  assert.equal(single.attention.outputGroupBytes, 0n);
  assert.equal(single.engram.broadcastTreeGroupBytesPerTP, 0n);
  assert.equal(cp.moeExpected.dispatchGroupBytes, single.moeExpected.dispatchGroupBytes * 16n);
  assert.throws(() => v41Step(config, { tokens: 10, nodeRanks: 4 }), /node-local TP/);
  assert.equal(v41Step(config, { tokens: 0 }).engram.idsGroupBytes, 0n);
});

test('Engram reverse responses follow owners and default BF16 wire, including asymmetric and self counts', () => {
  const counts = [[10, 2], [3, 20]];
  const bf16 = v41EngramRouting({ sendCounts: counts });
  assert.deepEqual(bf16.requests.rankSentBytesPerCall, [16n, 24n]);
  assert.deepEqual(bf16.responses.rankSentBytesPerCall, [1536n, 1024n]);
  assert.equal(bf16.responses.groupSentBytesPerCall, 2560n);
  assert.equal(v41EngramRouting({ sendCounts: counts, compressedInt8Wire: true }).responseBytesPerLookup, 288n);
  assert.equal(v41EngramRouting({ sendCounts: [[100]] }).responses.groupSentBytesPerCall, 0n);
});
