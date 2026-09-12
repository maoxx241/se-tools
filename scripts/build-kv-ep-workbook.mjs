import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { loadSpreadsheetRuntime } from '../lib/spreadsheet-runtime.mjs';
import { loadKvEpSpecs, KV_EP_DEFAULTS } from '../lib/kv-ep-specs.mjs';

const outputDir = path.resolve(process.argv[2] || 'outputs/kv-ep');
await fs.mkdir(outputDir, { recursive: true });
const { Workbook, SpreadsheetFile } = await loadSpreadsheetRuntime();
const wb = Workbook.create();
const summary = wb.worksheets.add('规格汇总'), kv = wb.worksheets.add('KV明细'), ep = wb.worksheets.add('EP通信');
const cases = await loadKvEpSpecs();
const value = (s,c,v) => { s.getRange(c).values=[[v]]; };
const formula = (s,c,f) => { s.getRange(c).formulas=[[`=${f}`]]; };
const header = (s,row,labels) => {
  const r=s.getRangeByIndexes(row-1,0,1,labels.length); r.values=[labels];
  r.format={fill:'#253D5B',font:{bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',rowHeight:40};
};
for(const s of [summary,kv,ep]) {
  s.showGridLines=false;
  s.getRange('A1:M65').format={font:{name:'Arial',size:10,color:'#202C3B'},rowHeight:25,verticalAlignment:'center'};
  s.getRange('A1:A65').format.columnWidth=19;
  s.getRange('B1:M65').format.columnWidth=15;
}
summary.tabColor='#253D5B';
value(summary,'A2','EP32 / EP256 · 通信量与 DP 域变化'); summary.getRange('A2').format.font={size:16,bold:true};
value(summary,'A3','通信单位 GiB/Step/EP组。每 DP 工作量不变，DP 扩大使总 token 数增加；已计入，不能再乘一次 DP。');
header(summary,5,['模型','EP','P TP','P DP','D TP','D DP','P组 tokens/Step','D组 tokens/Step','P AllToAllV合计','D MC2已建模量','P/Rank均值','D/Rank均值','部署条件']);
summary.getRange('G6:H15').setNumberFormat('#,##0'); summary.getRange('I6:L15').setNumberFormat('#,##0.000');
summary.getRange('M1:M65').format.columnWidth=33;
value(summary,'A18','配套 KV 存取 · 每请求 / 每 P DP 批 / P 整组'); summary.getRange('A18').format.font={size:14,bold:true};
value(summary,'A19','单位 GiB。存＝生成 KV 行＋最终辅助状态；取＝P→D 拉取。主干层，不含 draft/MTP 网络。');
header(summary,21,['模型','EP','P TP','P DP','D TP','D DP','存/请求','取/请求','存/P DP批','取/P DP批','存/P整组','取/P整组','部署条件']);
summary.getRange('G22:L31').setNumberFormat('#,##0.000');
value(summary,'A34','EP256 条件：DS V4 / Kimi 需补齐 128 个物理专家槽并验证 EPLB；未改模型 checkpoint。');
formula(summary,'A36','"P 整组批次：EP32 为 "&\'KV明细\'!$K$5*4&" 请求，EP256 为 "&\'KV明细\'!$K$5*32&" 请求；单请求 KV 不随 EP 改变。"');
value(summary,'A38','DS 存量含整个 prompt 生成的 SWA 行；阶段末常驻量见 KV 明细，不能按生成量配 HBM。');
value(summary,'A40','MC2 未计轮询读、重试和物理协议开销；不是整机 HBM 访存或 RoCE/HCCS 实测总量。');
value(summary,'A42','黄色输入可编辑；参数、公式与源码说明见 KV明细、EP通信及 analysis/KV-EP-SPECS.md。');

value(kv,'A2','KV 数据量 · 按单请求和全部 TP Rank'); kv.getRange('A2').format.font={size:16,bold:true};
value(kv,'A3','State 只记最终有效状态一次；不累计 Prefill 各 chunk 的状态覆盖写。SWA 拉取使用尾块上界。');
for(const [labelCell,inputCell,label,v] of [['A5','B5','Prompt tokens',262144],['D5','E5','Block tokens',128],['G5','H5','Spec conv slots',7],['J5','K5','Requests / P DP',16]]) {
  value(kv,labelCell,label); value(kv,inputCell,v); kv.getRange(inputCell).format.fill='#FFF0C2';
}
kv.getRange('E5').dataValidation={rule:{type:'list',values:['32','64','128']}};
kv.dataValidations.add({range:'B5',rule:{type:'whole',operator:'between',formula1:2,formula2:1048576}});
kv.dataValidations.add({range:'H5',rule:{type:'whole',operator:'between',formula1:0,formula2:64}});
kv.dataValidations.add({range:'K5',rule:{type:'whole',operator:'between',formula1:1,formula2:65536}});
header(kv,8,['模型','缓存组件','层数','每行 B/Rank','P 生成行','P 保留行','D 拉取行','P 副本/分片','D 副本/分片','存 GiB','保留 GiB','取 GiB']);
kv.getRange('B1:B65').format.columnWidth=43; kv.getRange('C9:I60').setNumberFormat('#,##0');
kv.getRange('J9:L60').setNumberFormat('#,##0.000000'); kv.freezePanes.freezeRows(8);
const totals=new Map(), kvCells=[]; let row=9;
for(const item of cases.filter(x=>x.ep===32)) {
  const start=row;
  for(const p of item.cache.components) {
    const N=item.profile==='glm53'?'$B$5':'($B$5-1)', B='$E$5';
    let g, keep, pull;
    if(p.component.startsWith('SWA')) {
      const W=Number(p.retainedRows); g=N; keep=`MIN(${N},${W})`;
      pull=`MIN(ROUNDUP(${N}/${B},0),ROUNDUP(${W}/${B},0)+1)*${B}`;
    } else if(p.component.includes('compressor state') || p.component.includes('indexer state')) {
      const W=Number(p.retainedRows), bs=p.component.startsWith('C128')?'($E$5/4)':'($E$5/16)';
      g=keep=`MIN(${N},${W})`; pull=`MIN(ROUNDUP(${N}/${bs},0),ROUNDUP(${W}/${bs},0)+1)*${bs}`;
    } else if(p.component.startsWith('Conv')) {
      g=keep=`MIN(${N},${p.generatedRows})`; pull=`${p.generatedRows}+$H$5`;
    } else if(p.component.startsWith('Recurrent')) { g=keep=pull='1';
    } else {
      const ratio=p.component.startsWith('C128')?128:p.component.startsWith('C4')?4:1;
      g=keep=`INT(${N}/${ratio})`; pull=`ROUNDUP(${N}/(${ratio}*${B}),0)*${B}`;
    }
    kv.getRange(`A${row}:D${row}`).values=[[item.model,p.component,Number(p.layers),Number(p.rowBytes)]];
    kv.getRange(`E${row}:G${row}`).formulas=[[`=${g}`,`=${keep}`,`=${pull}`]];
    kv.getRange(`H${row}:I${row}`).values=[[Number(p.prefillCopies),Number(p.decodeCopies)]];
    kv.getRange(`J${row}:L${row}`).formulas=[[`=C${row}*D${row}*E${row}*H${row}/2^30`,`=C${row}*D${row}*F${row}*H${row}/2^30`,`=C${row}*D${row}*G${row}*I${row}/2^30`]];
    kvCells.push({row,p}); row++;
  }
  value(kv,`A${row}`,item.model); value(kv,`B${row}`,'单请求合计');
  for(const col of ['J','K','L']) formula(kv,`${col}${row}`,`SUM(${col}${start}:${col}${row-1})`);
  kv.getRange(`A${row}:L${row}`).format.fill='#E8EEF5'; totals.set(item.slug,row); row+=2;
}
value(kv,`A${row+1}`,'实现：MooncakeConnector；P/D 同 block size、同 speculative 配置；无前缀命中、PCP/DCP=1。');
value(kv,`A${row+3}`,'GLM: 78 主干 / 21 Indexer；Kimi: 24 MLA + 69 KDA；Qwen: 23 GQA + 69 GDN。');
value(kv,`A${row+5}`,'来源：归档 config / VA 842b030f / vLLM c8438a3；已对照指定 main 1933f86；路径/哈希见源码清单。');

value(ep,'A2','EP 通信 · AllToAllV / A3 MC2 FullMesh'); ep.getRange('A2').format.font={size:16,bold:true};
value(ep,'A3','均衡路由；只计跨 Rank，排除自拷贝。EP=TP×DP。黄色输入可编辑。');
const inputs=[['P tokens/DP',16384],['D tokens/DP',128],['dispatch B',2],['scale B/token',0],['combine B',2],['histogram B',4]];
inputs.forEach(([label,n],i)=>{const c=i*2;ep.getRangeByIndexes(4,c,1,2).values=[[label,n]];ep.getRangeByIndexes(4,c+1,1,1).format.fill='#FFF0C2';});
ep.getRange('F5').dataValidation={rule:{type:'list',values:['1','2']}};
ep.getRange('H5').dataValidation={rule:{type:'list',values:['0','4']}};
value(ep,'A7','兼容性'); formula(ep,'B7','IF(AND(J5=2,OR(AND(F5=2,H5=0),AND(F5=1,H5=4))),"A3 packing supported","INVALID dtype/scale")');
ep.getRange('B7').conditionalFormats.add('containsText',{text:'INVALID',format:{fill:'#FDE9E7',font:{color:'#A32020',bold:true}}});
header(ep,9,['模型','EP','MoE层','H','TopK','逻辑专家','物理专家槽','补齐槽数','P TP','D TP','P tokens/Rank','D tokens/Rank','MC2 dispatch B']);
header(ep,23,['模型','EP','P dispatch GiB','P combine GiB','P计数AG GiB','P合计 GiB','D dispatch GiB','D计数 GiB','D combine GiB','D flags GiB','D合计 GiB','普通分支记录 B']);
ep.getRange('C24:K33').setNumberFormat('#,##0.000000'); ep.freezePanes.freezeRows(9);
for(let i=0;i<cases.length;i++) {
  const r=cases[i], a=10+i,b=24+i,s=22+i,e=6+i,k=totals.get(r.slug);
  ep.getRange(`A${a}:F${a}`).values=[[r.model,r.ep,r.facts.moeLayers,r.facts.H,r.facts.topK,r.facts.experts]];
  formula(ep,`G${a}`,`ROUNDUP(F${a}/B${a},0)*B${a}`); formula(ep,`H${a}`,`G${a}-F${a}`);
  ep.getRange(`I${a}:J${a}`).values=[[Number(r.topology.prefillTP),Number(r.topology.decodeTP)]];
  formula(ep,`K${a}`,`ROUNDUP($B$5/I${a},0)`);formula(ep,`L${a}`,`ROUNDUP($D$5/J${a},0)`);
  const packed=`ROUNDUP((ROUNDUP((ROUNDUP(D${a}*$F$5/32,0)*32+$H$5)/32,0)*32+32)/480,0)*512`;
  formula(ep,`M${a}`,packed);
  ep.getRange(`A${b}:B${b}`).formulas=[[`=A${a}`,`=B${a}`]];
  const RP=`C${a}*K${a}*E${a}*(B${a}-1)`,RD=`C${a}*L${a}*E${a}*(B${a}-1)`;
  const valid='AND($J$5=2,OR(AND($F$5=2,$H$5=0),AND($F$5=1,$H$5=4)))';
  const f=[`${RP}*(D${a}*$F$5+$H$5)`,`${RP}*D${a}*$J$5`,
    `IF(K${a}=0,0,C${a}*G${a}*$L$5*B${a}*(B${a}-1))`,null,
    `${RD}*M${a}`,`IF(L${a}=0,0,C${a}*G${a}*(B${a}-1)*32)`,`${RD}*D${a}*$J$5`,`${RD}*32`,null];
  f.forEach((x,j)=>{if(x) formula(ep,`${String.fromCharCode(67+j)}${b}`,`IF(${valid},(${x})/2^30,NA())`);});
  formula(ep,`F${b}`,`SUM(C${b}:E${b})`);formula(ep,`K${b}`,`SUM(G${b}:J${b})`);
  formula(ep,`L${b}`,`ROUNDUP((ROUNDUP(D${a}*$F$5/32,0)*32+$H$5)/32,0)*32+12`);
  summary.getRange(`A${s}:F${s}`).formulas=[[`='EP通信'!A${a}`,`='EP通信'!B${a}`,`='EP通信'!I${a}`,`=B${s}/C${s}`,`='EP通信'!J${a}`,`=B${s}/E${s}`]];
  summary.getRange(`G${s}:M${s}`).formulas=[[`='KV明细'!J${k}`,`='KV明细'!L${k}`,`=G${s}*'KV明细'!$K$5`,`=H${s}*'KV明细'!$K$5`,`=I${s}*D${s}`,`=J${s}*D${s}`,`=IF('EP通信'!H${a}>0,"需补齐128槽并验证EPLB","专家数整除；未实机验证")`]];
  summary.getRange(`A${e}:M${e}`).formulas=[[
    ...['A','B','C','D','E','F'].map(c=>`=${c}${s}`),`='EP通信'!K${a}*B${e}`,`='EP通信'!L${a}*B${e}`,
    `='EP通信'!F${b}`,`='EP通信'!K${b}`,`=I${e}/B${e}`,`=J${e}/B${e}`,`=M${s}`]];
}
ep.getRange('H10:H19').conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0,format:{fill:'#FFF0C2',font:{color:'#955700',bold:true}}});
value(ep,'A36','P 计数交换：histogram dtype 先设 4 B，Ring 为显式算法假设；采集实际 dtype/count 后可替换。');
value(ep,'A38','D dispatch = ceil((align32(align32(H×dtype)+scale)+32)/480)×512 B / 路由实例。');
value(ep,'A40','D combine 发送 H×2 B + 32 B 标志 / 实例；专家计数 32 B / 源 Rank / 远端专家。');
value(ep,'A42','普通 dispatch 分支在 L 列单列其 record；主表固定 FullMesh，不把 512 B 地址步长当发送长度。');
value(ep,'A44','来源：HCCL 170ddeec + ops-transformer f48a9346；只是所选分支的端点写量，非物理链路总量。');

function verify(expected) {
  for(let i=0;i<expected.length;i++) {
    const r=expected[i],n=22+i,m=6+i;
    const cells=[[`G${n}`,r.cache.generatedBytes],[`H${n}`,r.cache.pullBytes],[`I${n}`,r.batch.generatedBytes],[`J${n}`,r.batch.pullBytes],
      [`K${n}`,r.wholePrefillGroup.generatedBytes],[`L${n}`,r.wholePrefillGroup.pullBytes],[`I${m}`,r.prefillEP.modeledBytes],[`J${m}`,r.decodeEP.modeledRemoteWriteBytes]];
    for(const [c,b] of cells) assert(Math.abs(summary.getRange(c).values[0][0]*2**30-Number(b))<1, `${c} != CLI`);
    assert.equal(summary.getRange(`G${m}`).values[0][0],Number(r.stepWorkload.prefillTokensPerEPGroup));
    assert.equal(summary.getRange(`H${m}`).values[0][0],Number(r.stepWorkload.decodeTokensPerEPGroup));
  }
}
wb.recalculate(); verify(cases);
// Input-driven recalculation, crossing both context/block boundaries and quant packing.
value(kv,'B5',129);value(kv,'H5',0);value(kv,'K5',3);value(ep,'B5',8192);value(ep,'F5',1);value(ep,'H5',4);
wb.recalculate();verify(await loadKvEpSpecs({promptTokens:129,speculativeSlots:0,requestsPerDP:3,prefillTokensPerDP:8192,dispatchBytes:1,dispatchScaleBytes:4}));
value(kv,'B5',2);value(kv,'E5',64);value(ep,'B5',0);value(ep,'D5',0);
wb.recalculate();verify(await loadKvEpSpecs({promptTokens:2,blockSize:64,speculativeSlots:0,requestsPerDP:3,prefillTokensPerDP:0,decodeTokensPerDP:0,dispatchBytes:1,dispatchScaleBytes:4}));
value(kv,'B5',KV_EP_DEFAULTS.promptTokens);value(kv,'E5',128);value(kv,'H5',7);value(kv,'K5',16);value(ep,'B5',16384);value(ep,'D5',128);value(ep,'F5',2);value(ep,'H5',0);
wb.recalculate();verify(cases);
const scan=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:20},summary:'formula error scan'});
console.log(scan.ndjson);
for(const [sheetName,range,name] of [['规格汇总','A1:M16','ep-summary'],['规格汇总','A18:M42','kv-summary'],['KV明细',`A1:L${row+5}`,'kv-detail'],['EP通信','A1:M44','ep-detail']]) {
  const img=await wb.render({sheetName,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(outputDir,`${name}.png`),new Uint8Array(await img.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(outputDir,'kv-ep32-ep256.xlsx'));
console.log('Verified 10 cases against CLI, input mutations and 3 sheet renders; exported kv-ep32-ep256.xlsx');
