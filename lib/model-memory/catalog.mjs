import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { kimiMemorySpec } from './kimi.mjs';
import { evaluateMemory } from './engine.mjs';
import { validateMemorySpec, selectMemoryRows } from './schema.mjs';

export const memoryModels = ['kimi-k3'];
export const modelRoot = fileURLToPath(new URL('../../models/', import.meta.url));
export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

async function implementationHashes() {
  const calculation = await Promise.all(['engine.mjs', 'schema.mjs', 'kimi.mjs', 'catalog.mjs']
    .map(file => fs.readFile(new URL(file, import.meta.url))));
  return { calculationSha256: sha256(Buffer.concat(calculation)),
    workbookSchemaSha256: sha256(await fs.readFile(new URL('../../scripts/memory-workbook.mjs', import.meta.url))) };
}

function assemble(slug, spec, document, provenance) {
  validateMemorySpec(spec);
  if (document.schemaVersion !== 1 || !Array.isArray(document.scenarios) || !document.scenarios.length) throw new Error('Expected memory preset schemaVersion=1 and a non-empty scenarios array');
  const seen = new Set();
  const scenarios = document.scenarios.map(item => {
    if (typeof item.name !== 'string' || !item.name.length || item.name.length > 31 || /[\\/?*[\]:]/.test(item.name)
        || seen.has(item.name)) throw new Error(`Invalid or duplicate sheet name: ${item.name}`);
    seen.add(item.name);
    if (!['Prefill', 'Decode'].includes(item.phase) || !['language', 'full'].includes(item.scope)) throw new Error(`Invalid phase/scope: ${item.name}`);
    const preset = { ...item, inputs: { ...document.defaults, ...item.inputs }, partitions: item.partitions ?? [spec.facts.layers] };
    const rows = selectMemoryRows(spec, preset.recipe, preset.scope);
    if (!rows.length) throw new Error(`No tensors in scope: ${preset.scope}`);
    evaluateMemory(spec, rows, preset);
    return { preset, rows };
  });
  return { slug, spec, scenarios, provenance };
}

export async function loadMemoryModel(slug, { presetsPath } = {}) {
  if (!memoryModels.includes(slug)) throw new Error(`Unsupported memory model: ${slug}; supported: ${memoryModels.join(', ')}. Use --spec for a separate model description.`);
  const directory = path.join(modelRoot, slug);
  const configBytes = await fs.readFile(path.join(directory, 'config.json'));
  const presetBytes = await fs.readFile(presetsPath ?? path.join(directory, 'memory-presets.json'));
  const config = JSON.parse(configBytes);
  return { ...assemble(slug, kimiMemorySpec(config), JSON.parse(presetBytes),
    { configSha256: sha256(configBytes), presetsSha256: sha256(presetBytes), ...await implementationHashes() }), config };
}

export async function loadMemoryDescription(specPath, { presetsPath } = {}) {
  const sourceBytes = await fs.readFile(specPath), document = JSON.parse(sourceBytes);
  if (typeof document.id !== 'string' || !/^[a-z][a-z0-9-]{0,63}$/.test(document.id)) throw new Error('Model description id must be a safe lowercase slug');
  const presetBytes = presetsPath ? await fs.readFile(presetsPath) : sourceBytes;
  return assemble(document.id, document, presetsPath ? JSON.parse(presetBytes) : document,
    { descriptionSha256: sha256(sourceBytes), presetsSha256: sha256(presetBytes), ...await implementationHashes() });
}
