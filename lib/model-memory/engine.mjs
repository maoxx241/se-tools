export const GiB = 2 ** 30;
export const formats = {
  BF16: { bytes: 2 }, FP32: { bytes: 4 }, FP8: { bytes: 1 },
  MXFP4: { bytes: 0.5, group: 32, scale: 1 }, MXFP8: { bytes: 1, group: 32, scale: 1 },
  INT4: { bytes: 0.5, channel: 16 }, INT8: { bytes: 1, channel: 8 },
};
export const ruleInputs = {
  'KDA TP': 'tp', 'KDA O TP': 'kdaOTP', 'DSA TP': 'attentionTP', 'DSA O TP': 'attentionOTP',
  'LM head TP': 'lmHeadTP', 'Embedding TP': 'embeddingTP', 'Other TP': 'otherTP', 'Shared TP': 'sharedTP',
};
export const inputKeys = ['tp', 'kdaOTP', 'attentionTP', 'attentionOTP', 'lmHeadTP', 'embeddingTP', 'otherTP',
  'ep', 'sharedTP', 'dsaCP', 'dcp', 'tokens', 'blockSize', 'specTokens', 'convBytes', 'recurrentBytes', 'mlaDtype', 'indexDtype'];

export function storageBytes(shape, dtype) {
  const format = formats[dtype];
  if (!format) throw new Error(`Unsupported dtype: ${dtype}`);
  if (!shape.length || shape.some(n => !Number.isSafeInteger(n) || n < 1)) throw new Error(`Invalid local tensor shape: ${shape}`);
  const prefix = shape.slice(0, -1).reduce((a, b) => a * b, 1), last = shape.at(-1);
  const weight = prefix * Math.ceil(last * format.bytes);
  const scale = format.group ? prefix * Math.ceil(last / format.group) * format.scale : 0;
  const auxiliary = format.channel ? shape[0] * format.channel : 0;
  const total = weight + scale + auxiliary;
  if (!Number.isSafeInteger(total)) throw new Error('Tensor storage exceeds safe integer range');
  return { weight, scale, auxiliary, total };
}

export function stageOwnership(facts, partitions) {
  if (!partitions.length || partitions.some(n => !Number.isInteger(n) || n < 1)
      || partitions.reduce((a, b) => a + b, 0) !== facts.layers) throw new Error('PP partitions must contain every decoder layer exactly once');
  let start = 1;
  return partitions.map((count, stage) => {
    const end = start + count - 1, owns = ids => ids.filter(n => n >= start && n <= end).length;
    const value = { stage, start, end, layers: count, kda: owns(facts.kdaLayers), attention: owns(facts.attentionLayers),
      dense: owns(facts.denseLayers), moe: owns(facts.moeLayers), first: stage === 0, last: stage === partitions.length - 1 };
    start = end + 1;
    return value;
  });
}

export function stageRepeats(row, ownership, experts, ep) {
  if (row.placement) return (row.placement === 'first' ? ownership.first : ownership.last) ? row.instances ?? 1 : 0;
  const layers = row.layerIds.filter(n => n >= ownership.start && n <= ownership.end).length;
  return layers * (row.rule === 'EP' ? Math.ceil(experts / ep) : 1);
}

export function resolveInputs(preset, changes = {}) {
  const p = { ...preset.inputs, ...changes };
  for (const key of Object.keys(p)) if (!inputKeys.includes(key)) throw new Error(`Unknown deployment input: ${key}`);
  const links = { kdaOTP: 'tp', attentionOTP: 'tp', lmHeadTP: 'tp', embeddingTP: 'tp', otherTP: 'tp',
    dsaCP: 'tp', dcp: 'tp', ep: 'tp' };
  // Only declared preset links follow another input; sharedTP is a numeric default.
  for (const [name, source] of Object.entries(preset.links ?? {})) {
    if (!(name in changes)) {
      if (!(name in links) || source.key !== links[name]
        || !Number.isSafeInteger(source.multiplier ?? 1) || (source.multiplier ?? 1) < 1) throw new Error(`Unsupported input link: ${name}`);
      p[name] = p[source.key] * (source.multiplier ?? 1);
    }
  }
  if (!('attentionTP' in changes) && preset.attentionTPFromCP) p.attentionTP = p.dsaCP > 1 ? 1 : p.tp;
  return p;
}

export function validateInputs(spec, preset, p) {
  for (const key of ['tp', 'kdaOTP', 'attentionTP', 'attentionOTP', 'lmHeadTP', 'embeddingTP',
    'otherTP', 'ep', 'sharedTP', 'dsaCP', 'dcp', 'tokens', 'blockSize', 'convBytes', 'recurrentBytes']) {
    if (!Number.isSafeInteger(p[key]) || p[key] < 1) throw new Error(`Invalid positive integer input: ${key}`);
  }
  if (!Number.isInteger(p.specTokens) || p.specTokens < 0) throw new Error('Invalid speculative token count');
  if (p.tokens > spec.facts.maxTokens) throw new Error('Token count exceeds model context length');
  if (p.ep % p.tp || spec.facts.kdaHeads % p.tp || p.tp % p.dcp) throw new Error('Invalid TP/EP/DCP divisibility');
  if (preset.phase === 'Prefill' && p.dcp !== 1 || preset.phase === 'Decode' && p.dsaCP !== 1) throw new Error('Context parallelism is enabled in the wrong phase');
  if (!spec.facts.hasDSA && p.dsaCP !== 1) throw new Error('DSA CP requires an explicit DSA model adapter');
  if (p.dsaCP !== 1 && p.dsaCP !== p.tp) throw new Error('This DSA CP adapter supports CP1 or CP=TP');
  for (const key of ['mlaDtype', 'indexDtype']) if (!['BF16', 'INT8', 'FP8'].includes(p[key])) throw new Error(`Unsupported cache dtype: ${p[key]}`);
  if (!spec.facts.hasDSA && p.mlaDtype !== 'BF16') throw new Error('This non-DSA cache adapter supports BF16 only');
}

export function cacheBytes(facts, p, owner) {
  const localTokens = Math.ceil(p.tokens / (p.blockSize * p.dcp)) * p.blockSize;
  const latentPerToken = p.mlaDtype === 'BF16' ? facts.kvRank * 2 : facts.kvRank + Math.ceil(facts.kvRank / 128) * 4;
  const ropePerToken = facts.ropeDim * 2;
  const indexPerToken = !facts.hasDSA ? 0 : facts.indexDim * (p.indexDtype === 'BF16' ? 2 : 1)
    + (p.indexDtype === 'BF16' ? 0 : p.indexDtype === 'INT8' ? 2 : 4);
  const latent = localTokens * owner.attention * latentPerToken;
  const rope = localTokens * owner.attention * ropePerToken;
  const indexer = localTokens * p.dcp * owner.attention * indexPerToken;
  const recurrent = owner.kda * facts.kdaHeads / p.tp * facts.headDim ** 2 * p.recurrentBytes;
  const conv = owner.kda * 3 * facts.kdaHeads / p.tp * facts.headDim * (facts.convKernel - 1 + p.specTokens) * p.convBytes;
  const total = latent + rope + indexer + recurrent + conv;
  if (!Number.isSafeInteger(total)) throw new Error('Cache byte count exceeds safe integer range');
  return { localTokens, latentPerToken, ropePerToken, indexPerToken, latent, rope, indexer, recurrent, conv, total };
}

export function evaluateMemory(spec, rows, preset, changes = {}, dtypeChanges = {}) {
  const inputs = resolveInputs(preset, changes);
  validateInputs(spec, preset, inputs);
  const ownership = stageOwnership(spec.facts, preset.partitions);
  const stages = ownership.map(owner => {
    const weights = rows.map(row => {
      const divisor = row.rule in ruleInputs ? inputs[ruleInputs[row.rule]] : 1;
      const shape = row.shape.map((dim, axis) => axis === row.axis ? dim / divisor : dim);
      const dtype = dtypeChanges[`${row.group}/${row.name}`] ?? row.dtype;
      const stored = storageBytes(shape, dtype), global = storageBytes(row.shape, dtype);
      const repeat = stageRepeats(row, owner, spec.facts.experts, inputs.ep);
      if (!Number.isSafeInteger(stored.total * repeat) || !Number.isSafeInteger(global.total * row.repeat)) throw new Error('Repeated tensor bytes exceed safe integer range');
      return { name: `${row.group}/${row.name}`, dtype, divisor, shape, repeat, ...stored,
        bytes: stored.total * repeat, globalBytes: global.total * row.repeat };
    });
    const weight = weights.reduce((sum, r) => sum + r.bytes, 0);
    if (!Number.isSafeInteger(weight)) throw new Error('Stage weight bytes exceed safe integer range');
    const cache = cacheBytes(spec.facts, inputs, owner);
    const outputIndex = rows.findIndex(r => ['DSA MLA', 'Gated MLA'].includes(r.group) && r.name === 'o_proj');
    let buffer = 0;
    if (spec.facts.hasDSA && preset.phase === 'Prefill' && inputs.dsaCP > 1 && owner.attention) {
      const out = weights[outputIndex], row = rows[outputIndex];
      const full = storageBytes(row.shape, out.dtype);
      buffer = full.weight + full.scale;
      if (['BF16', 'FP32'].includes(out.dtype)) buffer += out.bytes;
    }
    return { ...owner, rows: weights, weight, cache, buffer, total: weight + cache.total + buffer };
  });
  return { inputs, stages, maxWeight: Math.max(...stages.map(s => s.weight)), maxCache: Math.max(...stages.map(s => s.cache.total)),
    maxTotal: Math.max(...stages.map(s => s.total)) };
}
