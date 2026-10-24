import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

export async function loadSpreadsheetRuntime() {
  const entry = process.env.ARTIFACT_TOOL_ENTRY || "@oai/artifact-tool";
  try {
    return await import(entry);
  } catch (cause) {
    if (!process.env.ARTIFACT_TOOL_ENTRY) {
      const bundled = path.join(homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs');
      if (existsSync(bundled)) return import(pathToFileURL(bundled).href);
    }
    throw new Error(`Cannot load spreadsheet runtime ${entry}. Provide @oai/artifact-tool or ARTIFACT_TOOL_ENTRY. The dependency-free export-analysis.mjs and archived Excel files remain usable.`, { cause });
  }
}
