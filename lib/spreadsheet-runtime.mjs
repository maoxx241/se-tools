export async function loadSpreadsheetRuntime() {
  const entry = process.env.ARTIFACT_TOOL_ENTRY || "@oai/artifact-tool";
  try {
    return await import(entry);
  } catch (cause) {
    throw new Error(`Cannot load spreadsheet runtime ${entry}. Provide @oai/artifact-tool or ARTIFACT_TOOL_ENTRY. The dependency-free export-analysis.mjs and archived Excel files remain usable.`, { cause });
  }
}
