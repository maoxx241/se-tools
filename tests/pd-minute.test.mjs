import test from 'node:test';
import assert from 'node:assert/strict';
import {minuteImpact,meanEventDelay,overlapProbability,loadMinuteSweep} from '../lib/pd-minute.mjs';
import {calculateStepTraffic,decodeTpTraffic} from '../lib/communication-requirements.mjs';
import {loadKvEpSpecs} from '../lib/kv-ep-specs.mjs';

const base={pullBytesPerRequest:1e9,ep:32,prefillTP:8,decodeTP:1,decodeBytesPerRank:1e8,
  layers:10,ttftSeconds:4,tpotMs:10,kvBandwidthGBps:100,decodeBandwidthGBps:100};
const close=(a,b,tol=1e-7)=>assert(Math.abs(a-b)<tol,`${a} != ${b}`);

test('finite-window event delay agrees with direct speed integration',()=>{
  for(const [c,w,C,d] of [[2,10,100,.1],[20,10,100,.4],[2,90,100,.7],[2,1,100,0]]) {
    let total=0;
    const n=100000;
    for(let i=0;i<n;i++) {
      let t=(i+.5)*C/n, remaining=c, elapsed=0;
      while(remaining>1e-10) {
        const phase=t%C, slow=phase<w, end=slow?w:C;
        const speed=slow?1-d:1,dt=Math.min(remaining/speed,end-phase);
        remaining-=dt*speed;t+=dt;elapsed+=dt;
      }
      total+=elapsed-c;
    }
    close(meanEventDelay(c,w,C,d),total/n,1e-6);
  }
});

test('layer geometry matches independent phase enumeration including short windows',()=>{
  for(const w of [.01,.5,4,40,99.9]) {
    const C=100,T=10,comm=2,L=5, c=comm/L,g=(T-comm)/L;
    let any=0,all=0;const n=100000;
    for(let i=0;i<n;i++) {
      const start=(i+.5)*C/n;
      let hit=false,full=true;
      for(let l=0;l<L;l++) {
        const s=start+l*T/L+g/2,e=s+c;
        let overlap=0;
        for(let k=0;k<2;k++)overlap+=Math.max(0,Math.min(e,k*C+w)-Math.max(s,k*C));
        hit||=overlap>1e-9;full&&=overlap>=c-1e-9;
      }
      any+=hit;all+=full;
    }
    close(overlapProbability(w,C,T,comm,L),any/n,2e-5);
    close(1-overlapProbability(C-w,C,T,comm,L),all/n,2e-5);
  }
});

test('minute rates, topology, TPOT/forward conversion and independent KV bandwidth',()=>{
  const a=minuteImpact(base),b=minuteImpact({...base,ep:256});
  assert.equal(a.status,'ok');assert.equal(a.kvBurstsPerMinute,15);
  assert.equal(b.requestHandoffsPerMinute,a.requestHandoffsPerMinute*8);
  assert.equal(b.kvPullBytesPerDecodeRank,a.kvPullBytesPerDecodeRank);
  assert.equal(b.requestsPerDecodeDPPerMinute,a.requestsPerDecodeDPPerMinute);
  const p=minuteImpact({...base,prefillGroups:2});
  close(p.windowMs,a.windowMs*2);
  const faster=minuteImpact({...base,kvBandwidthGBps:200});
  close(faster.windowMs,a.windowMs/2);
  const spec=minuteImpact({...base,outputTokensPerStep:4});
  assert.equal(spec.stepMs,40);close(spec.baselineStepsPerMinute,a.baselineStepsPerMinute/4);
  const split=minuteImpact({...base,burstsPerCycle:4});
  close(split.windowMs,a.windowMs/4);close(split.offeredKVUtilization,a.offeredKVUtilization);
  assert(split.partialStepsPerMinute>a.partialStepsPerMinute);
});

test('no transfer/loss, overload and infeasible timing do not produce plausible false results',()=>{
  const zero=minuteImpact({...base,pullBytesPerRequest:0});
  assert.equal(zero.overlappingStepsPerMinute,0);assert.equal(zero.extraMsForBaselineMinute,0);
  assert.equal(minuteImpact({...base,bandwidthLoss:0}).extraMsForBaselineMinute,0);
  assert.equal(minuteImpact({...base,exposedFraction:0}).extraMsForBaselineMinute,0);
  const slow=minuteImpact({...base,decodeBandwidthGBps:1});
  assert.equal(slow.status,'tpot-infeasible');assert.equal(slow.extraMsForBaselineMinute,null);
  const overloaded=minuteImpact({...base,kvBandwidthGBps:.001});
  assert.equal(overloaded.status,'kv-overload');assert.equal(overloaded.extraMsForBaselineMinute,null);
  assert.throws(()=>minuteImpact({...base,kvBandwidthGBps:0}));
});

test('144 scenarios contain 15/3/1 steady-state KV windows per minute and qualified V4.1',async()=>{
  const rows=await loadMinuteSweep();assert.equal(rows.length,144);
  assert.deepEqual([...new Set(rows.map(x=>x.result.kvBurstsPerMinute))],[15,3,1]);
  assert.equal(rows.filter(x=>x.slug==='deepseek-v4.1-flash').length,24);
  assert(rows.filter(x=>x.slug==='deepseek-v4.1-flash').every(x=>x.kvBasis.startsWith('planned')));
});

test('TP8 traffic is per rank, non-duplicated, and fixed when EP/DP scale',async()=>{
  for(const slug of ['kimi-k3','qwen3.8-2.4t-a95b']) {
    const specs=(await loadKvEpSpecs()).filter(x=>x.slug===slug);
    const totals=specs.map(m=>{
      const options={tp:8,ep:m.ep,spEnabled:true,dsa:false,sharedExpertTpEnabled:m.profile==='kimi'};
      const old=calculateStepTraffic(m.facts,128,128,options);
      const ledger=decodeTpTraffic(m.facts,128,8,options);
      assert.equal(ledger.totalBytesPerRank,Number(old.tpBoundary+old.sp+old.sharedExpertTp)/8);
      assert.equal(ledger.components.sharedExpert>0,m.profile==='kimi');
      return ledger.totalBytesPerRank;
    });
    assert(totals[0]>0);assert.equal(totals[0],totals[1]);
  }
  const rows=await loadMinuteSweep();
  assert(rows.filter(x=>x.inputs.decodeTP===1).every(x=>x.inputs.decodeTpBytesPerRank===0));
});

test('TP bandwidth and sharing independently affect total delay and feasibility',()=>{
  const input={...base,decodeTpBytesPerRank:2e8,tpBandwidthGBps:100};
  const both=minuteImpact(input),ep=minuteImpact(base);
  close(both.extraEpMsForBaselineMinute,ep.extraMsForBaselineMinute);
  close(both.extraMsForBaselineMinute,both.extraEpMsForBaselineMinute+both.extraTpMsForBaselineMinute);
  assert(both.extraTpMsForBaselineMinute>0);
  assert(both.overlappingStepsPerMinute>=ep.overlappingStepsPerMinute);
  close(both.overlappingStepsPerMinute,both.allCommunicationInsideStepsPerMinute+both.partialStepsPerMinute);
  close(minuteImpact({...input,tpSharedFraction:0}).extraMsForBaselineMinute,ep.extraMsForBaselineMinute);
  close(minuteImpact({...input,tpExposedFraction:0}).extraMsForBaselineMinute,ep.extraMsForBaselineMinute);
  assert(minuteImpact({...input,tpBandwidthGBps:50}).extraTpMsForBaselineMinute>both.extraTpMsForBaselineMinute);
  assert.equal(minuteImpact({...input,tpBandwidthGBps:20}).status,'tpot-infeasible');
  assert.throws(()=>minuteImpact({...input,tpBandwidthGBps:0}));
});
