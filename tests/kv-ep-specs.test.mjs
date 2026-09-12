import test from 'node:test';
import assert from 'node:assert/strict';
import { balancedEP, loadKvEpSpecs } from '../lib/kv-ep-specs.mjs';
import { estimateAllToAllV } from '../lib/collectives.mjs';

test('uniform routing reduces the HCCL off-diagonal count matrix, not DP twice', () => {
  const r = balancedEP({ep: 4, tokensPerRank: 8, layers: 2, hidden: 16, topK: 2, experts: 8});
  const actual = estimateAllToAllV({sendCounts: Array.from({length:4}, () => [64,64,64,64]), dtypeBytes:2, callsPerStep:2});
  assert.equal(r.alltoallv.dispatchPayload, actual.groupSentBytesPerStep);
  assert.equal(r.mc2.dispatchRecordBytes, 512n);
  assert.equal(r.mc2.combineFlagBytes, r.routes * 32n);
});
test('MC2 record packing distinguishes scale, flags, counts and window stride', () => {
  const r = balancedEP({ep:32, tokensPerRank:128, layers:1, hidden:7168, topK:6, experts:384});
  assert.equal(r.mc2.dispatchRecordBytes, 15360n);
  assert.equal(r.mc2.ordinaryDispatchRecordBytes, 14348n); // not its 512-aligned address stride
  const q = balancedEP({ep:32, tokensPerRank:128, layers:1, hidden:7168, topK:6, experts:384, dispatchBytes:1, dispatchScaleBytes:4});
  assert.equal(q.mc2.dispatchRecordBytes, 8192n);
  assert.throws(() => balancedEP({ep:32,tokensPerRank:1,layers:1,hidden:16,topK:1,experts:32,dispatchBytes:1}), /supports/);
});
test('ten cases preserve per-request KV and scale only the workload of the P group', async () => {
  const rows = await loadKvEpSpecs(); assert.equal(rows.length, 10);
  for (let i=0;i<10;i+=2) {
    assert.equal(rows[i].cache.generatedBytes, rows[i+1].cache.generatedBytes);
    assert.equal(rows[i].cache.pullBytes, rows[i+1].cache.pullBytes);
    assert.equal(rows[i+1].wholePrefillGroup.pullBytes, rows[i].wholePrefillGroup.pullBytes*8n);
    assert.equal(rows[i].wholePrefillGroup.requests,64n);
    assert.equal(rows[i+1].wholePrefillGroup.requests,512n);
    assert.equal(rows[i].stepWorkload.prefillTokensPerEPGroup,65536n);
    assert.equal(rows[i+1].stepWorkload.prefillTokensPerEPGroup,524288n);
    assert.equal(rows[i+1].stepWorkload.decodeTokensPerEPGroup,rows[i].stepWorkload.decodeTokensPerEPGroup*8n);
    // Per-DP load fixed: data payload grows by 255/31, not only 255/248.
    assert.equal(rows[i+1].prefillEP.dispatchPayload*31n,rows[i].prefillEP.dispatchPayload*255n);
  }
  for (const i of [1,3,7]) assert.equal(rows[i].redundantExpertsRequired,128n);
  assert.equal(rows[5].redundantExpertsRequired,0n);
});
test('PD pulls tail blocks / one state; generated rows differ from retained and transferred bytes', async () => {
  const rows=await loadKvEpSpecs(); const ds=rows[0].cache;
  assert.equal(ds.prefillTokens,262143n);
  assert.equal(ds.components[0].generatedRows,262143n);
  assert.equal(ds.components[0].retainedRows,128n);
  assert.equal(ds.components[0].pullRows,256n);
  assert.equal(ds.components[1].generatedRows,65535n);
  assert.equal(ds.components[1].pullRows,65536n);
  const kimi=rows[6].cache;
  assert.equal(kimi.components[1].generatedRows,3n);
  assert.equal(kimi.components[1].pullRows,10n);
  assert.equal(kimi.components[2].pullRows,1n);
  assert(kimi.pullBytes>kimi.generatedBytes);
  const glm=rows[4]; assert.equal(glm.facts.moeLayers,75);
  assert.equal(glm.cache.generatedBytes,glm.cache.pullBytes*8n);
  // KV-head replication at TP8 for four-head Qwen is intentional.
  assert.equal(rows[8].cache.components[0].rowBytes,1024n);
  const small=(await loadKvEpSpecs({promptTokens:129,speculativeSlots:0}))[6].cache;
  assert.equal(small.components[0].generatedRows,128n);
  assert.equal(small.components[0].pullRows,128n);
  assert.equal(small.components[1].pullRows,3n);
  const tiny=(await loadKvEpSpecs({promptTokens:2}))[0].cache;
  assert.equal(tiny.components[4].generatedRows,1n);
  assert.equal(tiny.components[5].generatedRows,1n);
});
