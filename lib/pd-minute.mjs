// Phase-averaged finite-window sensitivity. Not a scheduling/performance simulator.
import { loadKvEpSpecs } from './kv-ep-specs.mjs';

export const MINUTE_CASES = Object.freeze([
  { promptTokens: 16384, ttftSeconds: 4 },
  { promptTokens: 262144, ttftSeconds: 20 },
  { promptTokens: 1048576, ttftSeconds: 60 },
]);
export const MINUTE_DEFAULTS = Object.freeze({ requestsPerDP: 16, prefillGroups: 1,
  decodeGroups: 1, kvBandwidthGBps: 4800, decodeBandwidthGBps: 4800,
  decodeTokensPerDP: 128, outputTokensPerStep: 1, burstsPerCycle: 1,
  bandwidthLoss: .1, sharedFraction: 1, exposedFraction: 1 });

// Uniformly placed equal-length layer EP blocks. The complement consists of
// L-1 internal gaps plus the gap wrapping around the KV arrival period.
export function overlapProbability(windowMs, periodMs, stepMs, commMs, layers) {
  if (windowMs <= 0 || commMs === 0) return 0;
  if (windowMs >= periodMs) return 1;
  const c = commMs / layers, gap = (stepMs - commMs) / layers;
  const span = (layers - 1) * stepMs / layers + c;
  return 1 - ((layers - 1) * Math.max(0, gap - windowMs)
    + Math.max(0, periodMs - span - windowMs)) / periodMs;
}

// Integrate one event's completion delay over uniformly distributed START phase.
// Speed is 1 outside a fixed KV window and 1-d inside it. Unlike q*d/(1-d),
// this stops slowing the event when the KV window ends. c <= idle is required.
export function meanEventDelay(c, windowMs, periodMs, d) {
  if (c === 0 || windowMs === 0 || d === 0) return 0;
  if (windowMs >= periodMs) return c * d / (1-d);
  if (c > periodMs - windowMs + 1e-9) throw new Error('Event may cross multiple KV windows');
  const a = 1-d;
  return (c <= a*windowMs
    ? d*c*windowMs/a - d*d*c*c/(2*a*a)
    : d*c*windowMs + d*d*windowMs*windowMs/2) / periodMs;
}

export function minuteImpact({ pullBytesPerRequest, ep, prefillTP, decodeTP,
  decodeBytesPerRank, layers, ttftSeconds, tpotMs, ...options }) {
  const o = { ...MINUTE_DEFAULTS, ...options };
  for (const [k,v] of Object.entries({ pullBytesPerRequest, ep, prefillTP, decodeTP,
    decodeBytesPerRank, layers, ttftSeconds, tpotMs, ...o })) {
    if (!Number.isFinite(v)) throw new Error(`${k} must be finite`);
  }
  for (const k of ['requestsPerDP','prefillGroups','decodeGroups','burstsPerCycle','decodeTokensPerDP'])
    if (!Number.isInteger(o[k]) || o[k] <= 0) throw new Error(`${k} must be a positive integer`);
  for (const k of ['kvBandwidthGBps','decodeBandwidthGBps','outputTokensPerStep'])
    if (o[k] <= 0) throw new Error(`${k} must be positive`);
  for (const k of ['bandwidthLoss','sharedFraction','exposedFraction'])
    if (o[k]<0 || o[k]>1 || (k==='bandwidthLoss' && o[k]===1)) throw new Error(`Invalid ${k}`);
  if (pullBytesPerRequest < 0 || decodeBytesPerRank < 0 || ttftSeconds <= 0 || tpotMs <= 0
    || !Number.isInteger(layers) || layers <= 0 || ep <= 0 || prefillTP <= 0 || decodeTP <= 0)
    throw new Error('Invalid workload/topology');
  const pDP=ep/prefillTP, dDP=ep/decodeTP;
  const cycleMs=ttftSeconds*1000, periodMs=cycleMs/o.burstsPerCycle;
  const stepMs=tpotMs*o.outputTokensPerStep;
  const pBytes=o.requestsPerDP*pullBytesPerRequest/prefillTP/o.burstsPerCycle;
  const dBytes=pBytes*o.prefillGroups/o.decodeGroups;
  const windowMs=Math.max(pBytes,dBytes)/(o.kvBandwidthGBps*1e6);
  const commMs=decodeBytesPerRank/(o.decodeBandwidthGBps*1e6), layerCommMs=commMs/layers;
  let status='ok';
  if (windowMs>=periodMs) status='kv-overload';
  else if (commMs>stepMs+1e-9) status='tpot-infeasible';
  else if (stepMs>periodMs || layerCommMs>periodMs-windowMs) status='timeline-required';
  const baselineStepsPerMinute=60000/stepMs;
  const result={status, pDP, dDP, periodMs, stepMs, windowMs,
    kvBurstsPerMinute:60000/periodMs,
    requestHandoffsPerMinute:o.prefillGroups*pDP*o.requestsPerDP*60000/cycleMs,
    requestsPerDecodeDPPerMinute:o.prefillGroups*pDP*o.requestsPerDP*60000/cycleMs/(o.decodeGroups*dDP),
    kvSendBytesPerPrefillRank:pBytes, kvPullBytesPerDecodeRank:dBytes,
    offeredKVUtilization:windowMs/periodMs, commMs, layerCommMs,
    minimumDecodeBandwidthGBps:decodeBytesPerRank/(stepMs*1e6), baselineStepsPerMinute,
    overlappingStepsPerMinute:null, allEPInsideStepsPerMinute:null, partialStepsPerMinute:null,
    overlappingLayerCallsPerMinute:null, equivalentFullCommStepsPerMinute:null,
    extraMsForBaselineMinute:null, meanTpotIncreaseMs:null, tpotIncreaseFraction:null };
  if (status!=='ok') return result;
  const any=overlapProbability(windowMs,periodMs,stepMs,commMs,layers);
  const all=commMs===0 ? 0 : Math.max(0,1-overlapProbability(periodMs-windowMs,periodMs,stepMs,commMs,layers));
  const extra=baselineStepsPerMinute*layers*meanEventDelay(layerCommMs,windowMs,periodMs,o.bandwidthLoss)
    *o.sharedFraction*o.exposedFraction;
  return {...result,
    overlappingStepsPerMinute:baselineStepsPerMinute*any,
    allEPInsideStepsPerMinute:baselineStepsPerMinute*all,
    partialStepsPerMinute:baselineStepsPerMinute*Math.max(0,any-all),
    overlappingLayerCallsPerMinute:commMs===0||windowMs===0 ? 0 : baselineStepsPerMinute*layers*Math.min(1,(windowMs+layerCommMs)/periodMs),
    equivalentFullCommStepsPerMinute:baselineStepsPerMinute*windowMs/periodMs,
    extraMsForBaselineMinute:extra, meanTpotIncreaseMs:tpotMs*extra/60000,
    tpotIncreaseFraction:extra/60000 };
}

export async function loadMinuteSweep(options={}) {
  const {cases=MINUTE_CASES,tpots=[5,10,20,30],...overrides}=options;
  const o={...MINUTE_DEFAULTS,...overrides}, rows=[];
  for (const scenario of cases) {
    for (const r of await loadKvEpSpecs({promptTokens:scenario.promptTokens,
      requestsPerDP:o.requestsPerDP,decodeTokensPerDP:o.decodeTokensPerDP})) {
      for (const tpotMs of tpots) {
        const inputs={...o,pullBytesPerRequest:Number(r.cache.pullBytes??r.cache.plannedPullBytes),
          ep:r.ep,prefillTP:Number(r.topology.prefillTP),decodeTP:Number(r.topology.decodeTP),
          decodeBytesPerRank:Number(r.decodeEP.modeledRemoteWriteBytes)/r.ep,
          layers:r.facts.moeLayers,ttftSeconds:scenario.ttftSeconds,tpotMs};
        rows.push({model:r.model,slug:r.slug,profile:r.profile,promptTokens:scenario.promptTokens,
          placementStatus:r.placementStatus,kvBasis:r.cache.pullBytes===null?'planned; PD unverified':'source-derived tensor transfer',
          inputs,result:minuteImpact(inputs)});
      }
    }
  }
  return rows;
}
