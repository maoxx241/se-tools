import { formats, ruleInputs } from './engine.mjs';

const groups = new Set(['Embedding / Output', 'Decoder common', 'Dense MLP', 'KDA', 'Gated MLA',
  'DSA MLA', 'DSA Indexer', 'MoE', 'ViT patch', 'ViT blocks', 'ViT final', 'MM projector']);
const positive = n => Number.isSafeInteger(n) && n > 0;

export function validateMemorySpec(spec) {
  if (!spec || !spec.facts || !Array.isArray(spec.rows) || !spec.rows.length) throw new Error('Model description requires facts and rows');
  const f = spec.facts;
  for (const key of ['layers', 'headDim', 'convKernel', 'maxTokens']) if (!positive(f[key])) throw new Error(`Invalid model fact: ${key}`);
  for (const key of ['experts', 'sharedExperts', 'kdaHeads', 'attentionHeads', 'kvRank', 'ropeDim', 'indexDim', 'indexHeads', 'visionLayers']) {
    if (!Number.isSafeInteger(f[key]) || f[key] < 0) throw new Error(`Invalid nonnegative model fact: ${key}`);
  }
  for (const pair of [['kdaLayers', 'attentionLayers'], ['denseLayers', 'moeLayers']]) {
    const ids = pair.flatMap(key => {
      if (!Array.isArray(f[key])) throw new Error(`Missing layer list: ${key}`);
      return f[key];
    });
    if (ids.length !== f.layers || new Set(ids).size !== f.layers
      || ids.some(n => !positive(n) || n > f.layers)) throw new Error(`${pair.join('/')} must partition all layers (1-based)`);
  }
  if (typeof f.hasDSA !== 'boolean' || f.hasDSA && (!f.indexDim || !f.indexHeads)) throw new Error('Invalid DSA cache facts');
  if (f.kdaLayers.length && !f.kdaHeads) throw new Error('KDA layers require KDA heads');
  if (!Array.isArray(spec.recipes) || !spec.recipes.length || spec.recipes.some(name => typeof name !== 'string')) throw new Error('Model description requires named recipes');
  const seen = new Set();
  for (const row of spec.rows) {
    const key = `${row.group}/${row.name}`;
    if (!groups.has(row.group) || typeof row.name !== 'string' || !row.name || seen.has(key)) throw new Error(`Invalid or duplicate tensor: ${key}`);
    seen.add(key);
    if (!Array.isArray(row.shape) || !row.shape.length || !row.shape.every(positive)) throw new Error(`Invalid shape: ${key}`);
    if (!formats[row.dtype]) throw new Error(`Unsupported tensor dtype: ${row.dtype}`);
    if (!['Replicated', 'EP', ...Object.keys(ruleInputs)].includes(row.rule)) throw new Error(`Unsupported partition rule: ${row.rule}`);
    if (!Array.isArray(row.layerIds) || new Set(row.layerIds).size !== row.layerIds.length
      || row.layerIds.some(id => !positive(id) || id > f.layers)) throw new Error(`Invalid tensor layers: ${key}`);
    if (row.placement && (!['first', 'last'].includes(row.placement) || row.layerIds.length)) throw new Error(`Invalid endpoint placement: ${key}`);
    if (!row.placement && !row.layerIds.length) throw new Error(`Missing layer/endpoint ownership: ${key}`);
    const layerKind = row.group === 'KDA' ? f.kdaLayers
      : ['Gated MLA', 'DSA MLA', 'DSA Indexer'].includes(row.group) ? f.attentionLayers
      : row.group === 'Dense MLP' ? f.denseLayers : row.group === 'MoE' ? f.moeLayers : null;
    if (layerKind && row.layerIds.some(id => !layerKind.includes(id))) throw new Error(`Tensor layers disagree with architecture: ${key}`);
    if (row.instances != null && !positive(row.instances)) throw new Error(`Invalid instance count: ${key}`);
    if (row.rule === 'EP' && !f.experts) throw new Error(`EP tensor requires experts: ${key}`);
    if (row.axis != null && (!Number.isInteger(row.axis) || row.axis < 0 || row.axis >= row.shape.length)) throw new Error(`Invalid sharding axis: ${key}`);
    for (const [recipe, dtype] of Object.entries(row.recipeDtypes ?? {})) {
      if (!spec.recipes.includes(recipe) || !formats[dtype]) throw new Error(`Invalid recipe override: ${key}/${recipe}`);
    }
  }
  if (f.hasDSA && f.attentionLayers.length && !seen.has('DSA MLA/o_proj')) throw new Error('DSA adapter requires an explicit o_proj buffer tensor');
  return spec;
}

export function selectMemoryRows(spec, recipe, scope = 'full') {
  if (!spec.recipes.includes(recipe)) throw new Error(`Unknown quantization recipe: ${recipe}`);
  return spec.rows.filter(r => scope === 'full' || !r.group.startsWith('ViT') && r.group !== 'MM projector').map(row => ({
    ...row, dtype: row.recipeDtypes?.[recipe] ?? row.dtype, shared: row.rule === 'Shared TP',
    repeat: row.layerIds.length ? row.layerIds.length * (row.rule === 'EP' ? spec.facts.experts : 1) : row.instances ?? 1,
    axis: row.axis ?? (row.rule.includes('O TP') || /down_proj/.test(row.name) && ['Shared TP', 'Other TP'].includes(row.rule) ? 1 : 0),
  }));
}
