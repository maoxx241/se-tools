import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { loadSpreadsheetRuntime } from '../lib/spreadsheet-runtime.mjs';
import { models,modelSlugs,readModelConfig } from '../lib/model-catalog.mjs';
import { modelSpec } from './model-analysis-specs.mjs';
import { loadKvEpSpecs } from '../lib/kv-ep-specs.mjs';
import { loadContentionSweep } from '../lib/pd-contention.mjs';
import { findCommunication,addModelContention,verifyModelContention,addSweepContention } from './contention-workbook.mjs';

const mode=process.argv[2]||'--inspect';assert(['--inspect','--write'].includes(mode));
const root=path.resolve(process.argv[3]||'outputs/pd-contention');
await fs.mkdir(root,{recursive:true});
const {FileBlob,SpreadsheetFile}=await loadSpreadsheetRuntime();
const rows=[];
async function render(wb,sheetName,range,file) {
  const img=await wb.render({sheetName,range,scale:1.2,format:'png'});
  await fs.writeFile(path.join(root,file),new Uint8Array(await img.arrayBuffer()));
}
function sameCells(before,after) {
  before.forEach((row,i)=>row.forEach((x,j)=>{
    const y=after[i][j];
    if(typeof x==='number' && typeof y==='number')assert(Math.abs(x-y)<=Math.max(1e-8,Math.abs(x)*1e-12),`Changed original ${i},${j}: ${x} -> ${y}`);
    else assert.equal(x??'',y??'',`Changed original ${i},${j}`);
  }));
}
for(const model of models) {
  const slug=modelSlugs[model.key],file=`models/${slug}/${slug}-analysis.xlsx`;
  const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(file));
  const facts=modelSpec(model,await readModelConfig(slug)).facts;
  for(const phase of ['Prefill','Decode']) {
    const sheet=wb.worksheets.getItem(phase),comm=findCommunication(sheet);
    if(mode==='--inspect') {
      if(phase==='Prefill')await render(wb,phase,`A${comm.first-1}:P${Math.min(comm.last,comm.first+7)}`,`${slug}-before.png`);
      console.log(`${slug} ${phase}: communication ${comm.first}:${comm.last}, total ${comm.total}`);
      continue;
    }
    const old=sheet.getRange(`A1:P${Math.max(comm.total,65)}`),before=old.values,formulas=old.formulas;
    const region=addModelContention(sheet,phase,facts,model.profile);
    const result=verifyModelContention(wb,sheet,region);
    sameCells(before,old.values);assert.deepEqual(old.formulas,formulas);
    const {comm:unused,...coordinates}=region;
    rows.push({slug,model:model.key,phase,...result,region:coordinates,scope:'current workbook inputs; listed communications only; normalized 100 GB/s, 10% loss, full coverage/exposure'});
    await render(wb,phase,`A${region.start}:L${region.total}`,`${slug}-${phase}.png`);
  }
  if(mode==='--write') {
    wb.recalculate();
    const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:10}});
    assert(!/"kind":"match"/.test(errors.ndjson),errors.ndjson);
    const dest=path.join(root,file);await fs.mkdir(path.dirname(dest),{recursive:true});
    await(await SpreadsheetFile.exportXlsx(wb)).save(dest);console.log(`Updated ${file}`);
  }
}
if(mode==='--write') {
  const file='examples/kv-ep-sweep/kv-ep32-ep256.xlsx';
  const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(file)),sheet=wb.worksheets.getItem('规格汇总');
  const before=sheet.getRange('A1:M48').values,formulas=sheet.getRange('A1:M48').formulas;
  const cases=await loadKvEpSpecs();const region=addSweepContention(sheet,cases);
  wb.recalculate();sameCells(before,sheet.getRange('A1:M48').values);assert.deepEqual(formulas,sheet.getRange('A1:M48').formulas);
  const expected=await loadContentionSweep();
  for(let i=0;i<expected.length;i++)for(const [c,n] of [['C',expected[i].prefill.baselineBandwidthMs],['E',expected[i].prefill.deltaForwardMs],['F',expected[i].decode.baselineBandwidthMs],['H',expected[i].decode.deltaForwardMs]])
    assert(Math.abs(sheet.getRange(`${c}${58+i}`).values[0][0]-n)<1e-7);
  sheet.getRange('M58').values=[[50]];sheet.getRange('N69').values=[[200]];wb.recalculate();
  assert(Math.abs(sheet.getRange('E58').values[0][0]-expected[0].prefill.deltaForwardMs*2)<1e-7);
  assert(Math.abs(sheet.getRange('H69').values[0][0]-expected[11].decode.deltaForwardMs/2)<1e-7);
  sheet.getRange('M58').formulas=[['=$B$55']];sheet.getRange('N69').formulas=[['=$D$55']];wb.recalculate();
  await render(wb,'规格汇总',`A51:O${region.last+4}`,'ep32-ep256-contention.png');
  const dest=path.join(root,file);await fs.mkdir(path.dirname(dest),{recursive:true});await(await SpreadsheetFile.exportXlsx(wb)).save(dest);
  await fs.writeFile(path.join(root,'model-timing-results.json'),JSON.stringify(rows,null,2)+'\n');
  await fs.writeFile(path.join(root,'ep-timing-results.json'),JSON.stringify(expected,(_,v)=>typeof v==='bigint'?v.toString():v,2)+'\n');
  console.log('Verified 14 phase extensions, preserved original cells/formulas, and 12 EP cases.');
}
