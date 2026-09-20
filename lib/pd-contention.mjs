// Bandwidth sensitivity, not an NPU performance simulator.
import { loadKvEpSpecs } from './kv-ep-specs.mjs';

export const CONTENTION_DEFAULTS = Object.freeze({ bandwidthGBps: 100, bandwidthLoss: .1,
  sharedFraction: 1, contentionCoverage: 1, exposedFraction: 1, baselineForwardMs: null });

export function contentionImpact(bytesPerRank, options = {}) {
  const o = { ...CONTENTION_DEFAULTS, ...options }, bytes = Number(bytesPerRank);
  if (!Number.isFinite(bytes) || bytes < 0) throw new Error('bytesPerRank must be finite and nonnegative');
  if (!Number.isFinite(o.bandwidthGBps) || o.bandwidthGBps <= 0) throw new Error('bandwidthGBps must be positive');
  for (const k of ['bandwidthLoss', 'sharedFraction', 'contentionCoverage', 'exposedFraction']) {
    if (!Number.isFinite(o[k]) || o[k] < 0 || o[k] > 1 || (k === 'bandwidthLoss' && o[k] === 1))
      throw new Error(`Invalid ${k}`);
  }
  const baselineBandwidthMs = bytes / (o.bandwidthGBps * 1e6);
  const deltaCommunicationMs = baselineBandwidthMs * o.sharedFraction * o.contentionCoverage * o.bandwidthLoss / (1-o.bandwidthLoss);
  const deltaForwardMs = deltaCommunicationMs * o.exposedFraction;
  const f = o.baselineForwardMs;
  if (f !== null && (!Number.isFinite(f) || f <= 0)) throw new Error('baselineForwardMs must be positive or null');
  if (f !== null && f < baselineBandwidthMs * o.exposedFraction)
    throw new Error('baselineForwardMs is shorter than the assumed exposed communication time');
  return { baselineBandwidthMs, contendedBandwidthMs: baselineBandwidthMs + deltaCommunicationMs,
    deltaCommunicationMs, deltaForwardMs, baselineForwardMs: f,
    contendedForwardMs: f === null ? null : f + deltaForwardMs,
    forwardSlowdown: f === null ? null : deltaForwardMs / f,
    fixedWorkThroughputLoss: f === null ? null : deltaForwardMs / (f + deltaForwardMs) };
}

export async function loadContentionSweep(options = {}) {
  const cases = await loadKvEpSpecs(options.kvOptions || {});
  return cases.map(r => ({ model:r.model, slug:r.slug, ep:r.ep, topology:r.topology,
    placementStatus:r.placementStatus, kvTransferStatus:r.cache.transferStatus || 'source estimate; no PD timing measurement',
    assumptions:{...CONTENTION_DEFAULTS,...options}, scope:'MoE EP only; per-rank mean endpoint bytes; bandwidth term per Step',
    prefill:contentionImpact(Number(r.prefillEP.modeledBytes)/r.ep,options),
    decode:contentionImpact(Number(r.decodeEP.modeledRemoteWriteBytes)/r.ep,options) }));
}
