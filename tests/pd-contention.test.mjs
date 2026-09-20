import test from 'node:test';
import assert from 'node:assert/strict';
import { contentionImpact,loadContentionSweep } from '../lib/pd-contention.mjs';

test('10% less bandwidth adds 1/9 communication time, not 10% forward time',()=>{
  const r=contentionImpact(1e9,{baselineForwardMs:50});
  assert.equal(r.baselineBandwidthMs,10);
  assert(Math.abs(r.deltaCommunicationMs-10/9)<1e-12);
  assert(Math.abs(r.forwardSlowdown-1/45)<1e-12);
  assert.equal(contentionImpact(1e9).forwardSlowdown,null);
  assert.equal(contentionImpact(1e9).contendedForwardMs,null);
});
test('shared resources, contention coverage and exposed critical path are independent',()=>{
  const r=contentionImpact(1e9,{sharedFraction:.5,contentionCoverage:.4,exposedFraction:.25});
  assert(Math.abs(r.deltaForwardMs-(10/9*.5*.4*.25))<1e-12);
  for(const k of ['bandwidthLoss','sharedFraction','contentionCoverage','exposedFraction'])
    assert.equal(contentionImpact(1e9,{[k]:0}).deltaForwardMs,0);
  assert.equal(contentionImpact(0).deltaForwardMs,0);
  assert.throws(()=>contentionImpact(1,{bandwidthGBps:0}),/positive/);
  assert.throws(()=>contentionImpact(1,{bandwidthLoss:1}),/Invalid/);
  assert.throws(()=>contentionImpact(1e9,{baselineForwardMs:1}),/shorter/);
});
test('EP mean time divides by rank count before using per-rank bandwidth',async()=>{
  const rows=await loadContentionSweep();assert.equal(rows.length,12);
  for(let i=0;i<rows.length;i+=2) {
    assert(rows[i+1].prefill.deltaForwardMs>rows[i].prefill.deltaForwardMs);
    assert(rows[i+1].prefill.deltaForwardMs<rows[i].prefill.deltaForwardMs*1.1);
  }
});
