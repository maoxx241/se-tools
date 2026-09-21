import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { loadSpreadsheetRuntime } from '../lib/spreadsheet-runtime.mjs';
import { readModelConfig } from '../lib/model-catalog.mjs';
import { loadPrefixModels, engramGeometry, engramTransfer, prefixTransfer, hostTransferReport } from '../lib/host-transfers.mjs';

const out=path.resolve(process.argv[2]||'outputs/host-transfers');
await fs.mkdir(out,{recursive:true});
const {Workbook,SpreadsheetFile,FileBlob}=await loadSpreadsheetRuntime();
const wb=Workbook.create(), e=wb.worksheets.add('Engram H2D'), p=wb.worksheets.add('Prefix RH2D');
const models=await loadPrefixModels(), geometry=engramGeometry(await readModelConfig('deepseek-v4.1-flash'));
const val=(s,c,v)=>{s.getRange(c).values=[[v]];};
const f=(s,c,v)=>{s.getRange(c).formulas=[['='+v]];};
const input=(s,c,v)=>{val(s,c,v);s.getRange(c).format.fill='#FFF2CC';};
const note=(s,r,text,end='L')=>{s.getRange(`A${r}:${end}${r}`).merge();val(s,`A${r}`,text);s.getRange(`A${r}:${end}${r}`).format={wrapText:true,rowHeight:30};};
const title=(s,r,text)=>{note(s,r,text);s.getRange(`A${r}`).format.font={name:'Carlito',size:14,bold:true};};
const header=(s,r,labels)=>{
  const range=s.getRangeByIndexes(r-1,0,1,labels.length);range.values=[labels];
  range.format={fill:'#D9E1F2',font:{name:'Carlito',size:11,bold:true,color:'#000000'},wrapText:true,rowHeight:48,
    horizontalAlignment:'center',verticalAlignment:'center',borders:{preset:'all',style:'thin',color:'#A6A6A6'}};
};
const label=(s,cells,text)=>{s.getRange(cells).merge();val(s,cells.split(':')[0],text);s.getRange(cells).format.wrapText=true;};
for(const s of [e,p]){
  s.showGridLines=true;s.tabColor='#D9E1F2';
  s.getRange('A1:N210').format={font:{name:'Carlito',size:11,color:'#000000'},rowHeight:23,verticalAlignment:'center'};
  s.getRange('A:A').format.columnWidth=23;s.getRange('B:L').format.columnWidth=15;
  s.getRange('M:M').format.columnWidth=16;s.getRange('N:N').format.columnWidth=44;
  s.getRange('B1:M210').setNumberFormat('#,##0');
  s.freezePanes.freezeRows(11);
}
title(e,2,'Engram offloading · P / D H2D');
note(e,3,'黄色可编辑。主表按 V4.1 token lookup；CPU 表 FP8/MXFP8 → CPU 解量化 → BF16 H2D。单卡列为均衡 owner 平均值。');
for(const [r,a,b,c] of [[5,'Requests / P DP','P chunk tokens / DP','D tokens / DP / step'],[6,'P TP','D TP','Ranks / node'],[7,'CPU-served vectors','H2D bytes / element','H2D vector format'],[8,'Device metadata 0/1','Lookups / token','Vector bytes / token']]){
  label(e,`A${r}:C${r}`,a);label(e,`E${r}:G${r}`,b);label(e,`I${r}:K${r}`,c);
  e.getRange(`A${r}:L${r}`).format.rowHeight=29;
}
f(e,'D5',"'Prefix RH2D'!D5");input(e,'H5',16384);input(e,'L5',128);
input(e,'D6',8);input(e,'H6',1);input(e,'L6',8);
input(e,'D7',1);e.getRange('D7').setNumberFormat('0%');input(e,'H7',2);
f(e,'L7','IF(H7=2,"BF16","custom bytes")');input(e,'D8',1);
f(e,'H8','F36');f(e,'L8','G36');
note(e,9,'Requests、长度、EP 共用第二页输入；H2D total = vectors + int64 IDs/order + device metadata。设备间 AllToAll / TP broadcast 不计入 H2D。');
header(e,11,['Stage / batch','EP','TP','DP','Tokens / DP','Rounds / DP','Vectors MiB / rank','IDs + meta MiB / rank','H2D total MiB / rank','H2D total GiB / node','H2D total GiB / EP','TP leader H2D MiB']);
const engramRows=[];
for(let group=0;group<2;group++){
  const er=63+group;
  for(let k=0;k<5;k++){
    const r=12+group*5+k, prefill=k!==1;
    val(e,`A${r}`,k===0?'P forward':k===1?'D forward':`P batch · case ${k-1}`);
    f(e,`B${r}`,`'Prefix RH2D'!D${er}`);f(e,`C${r}`,prefill?'$D$6':'$H$6');f(e,`D${r}`,`B${r}/C${r}`);
    f(e,`E${r}`,k===0?'$H$5':k===1?'$L$5':`$D$5*'Prefix RH2D'!B${k+61}`);
    f(e,`F${r}`,k<2?'1':`ROUNDUP(E${r}/$H$5,0)`);
    f(e,`G${r}`,`E${r}/C${r}*$L$8*$D$7/2^20`);
    f(e,`H${r}`,`(E${r}/C${r}*$H$8*16+F${r}*$B$36*($L$6+1)*8*$D$8)/2^20`);
    f(e,`I${r}`,`G${r}+H${r}`);f(e,`J${r}`,`I${r}*$L$6/1024`);f(e,`K${r}`,`I${r}*B${r}/1024`);
    f(e,`L${r}`,`G${r}+(E${r}*$H$8*16+F${r}*$B$36*($L$6+1)*8*$D$8)/2^20`);
    engramRows.push({r,group,k});
  }
}
e.getRange('G12:L21').setNumberFormat('#,##0.000000');
note(e,23,'P batch 是每 P DP 整批完整 prompt 的累计搬运；P/D forward 是一次调度 step。默认没有 lookup 去重，重复命中的向量仍逐次搬运。');
note(e,24,'CPU-served vectors 默认 100%。降低此值仅模拟额外的 HBM 向量缓存/混合驻留；当前 offload 路径没有这一命中缓存。');
note(e,25,'TP leader 额外上传查询 IDs 和排序下标；所有 rank 分摊向量 owner 负载。双缓冲只增加暂存空间，不使传输量乘 2。');
note(e,26,'单位：MiB=2^20 B，GiB=2^30 B。反向 D2H 的查询 IDs、metadata，以及 CPU 内部解量化访存不混入本页 H2D。');
title(e,28,'Model inputs · 下列参数可直接修改');
header(e,30,['Model','Engram layers','Max n-gram','Heads / n-gram','Vector dim','Lookups / token','Vector B / token','CPU storage','H2D format','Scope']);
for(let i=0;i<models.length;i++){
  const r=31+i,m=models[i],isV41=m.profile==='deepseek_v41';
  val(e,`A${r}`,m.name);
  for(const [col,v] of [['B',isV41?geometry.layers:0],['C',isV41?geometry.ngram:1],['D',isV41?geometry.heads:0],['E',isV41?geometry.width:0]])input(e,`${col}${r}`,v);
  f(e,`F${r}`,`B${r}*MAX(0,C${r}-1)*D${r}`);f(e,`G${r}`,`F${r}*E${r}*$H$7`);
  val(e,`H${r}`,isV41?'FP8 / MXFP8':'—');f(e,`I${r}`,isV41?'$L$7':'"—"');
  e.getRange(`J${r}:L${r}`).merge();val(e,`J${r}`,isV41?'Rows 12–21 use this model':'No Engram in archived config');
}
note(e,38,'CPU FP8 storage is 256 codes + 8 E8M0 scales per row; current H2D is 256 × 2 = 512 B per lookup. Source snapshot: 2026-09-22.');
note(e,40,'Source: GDzhu01/vllm-ascend-v41-private @ 6bb7aee · engram_hbm.py (lookup_local / _forward_with_gathered / route_many).');
note(e,41,'https://github.com/GDzhu01/vllm-ascend-v41-private/blob/6bb7aeecf01b660ae14d9adddfc2f7035d3d1305/vllm_ascend/models/deepseek_v41/engram_hbm.py');
note(e,42,'Config: models/deepseek-v4.1-flash/config.json · HF DeepSeek-V4.1-Flash @ dba1be0a40aa45a94ad051997016db3960a90277');

title(p,2,'Prefix pooling · D host → P device RH2D');
note(p,3,'D 节点 host pool 中的前缀拉到 P。六模型、EP32/256、16K/256K/1M；黄色可编辑。按接收端 payload 计一次，不再把发送和接收相加。');
for(const [r,a,b,c] of [[5,'Requests / P DP','Prefix hit %','P-local reuse % requests'],[6,'Base block tokens','Extra conv slots','Model scope']]){
  label(p,`A${r}:C${r}`,a);label(p,`E${r}:G${r}`,b);label(p,`I${r}:K${r}`,c);p.getRange(`A${r}:L${r}`).format.rowHeight=32;
}
input(p,'D5',16);input(p,'H5',1);input(p,'L5',0);p.getRange('H5').setNumberFormat('0%');p.getRange('L5').setNumberFormat('0%');
input(p,'D6',128);input(p,'H6',0);val(p,'L6','Target only');
note(p,8,'默认每个请求的前缀均需远端拉取；P-local reuse 是整请求已在 P 命中的比例。Kimi / Qwen 需 align state；P/D TP 与缓存布局须兼容。');
note(p,9,'G–I 是一个远端命中请求的量；J–L 是整批实际 RH2D。EP=TP×DP。P TP 副本计入流量，D 的重复副本不重复相加。');
header(p,11,['Model','EP','P TP','P DP','Input tokens','Pool-hit tokens','Cache MiB / rank / req','State MiB / rank / req','Total MiB / all P TP / req','RH2D GiB / rank / batch','RH2D GiB / P DP / batch','RH2D GiB / P EP / batch']);
title(p,50,'Topology & block alignment · P/D 参数与池化对齐');
header(p,53,['Model','P TP','D TP','Pool align tokens','Base block','Input check','Cache / transfer rule']);
for(let i=0;i<models.length;i++){
  const r=54+i,m=models[i];val(p,`A${r}`,m.name);input(p,`B${r}`,m.prefillTP);input(p,`C${r}`,m.decodeTP);
  f(p,`D${r}`,`$D$6*${m.poolAlignment/128}`);p.getRange(`D${r}`).format.fill='#FFF2CC';f(p,`E${r}`,'$D$6');
  p.getRange(`G${r}:L${r}`).merge();val(p,`G${r}`,m.note);p.getRange(`G${r}:L${r}`).format.wrapText=true;p.getRange(`A${r}:L${r}`).format.rowHeight=33;
}
header(p,62,['Case','Input tokens','EP case','EP']);
for(let i=0;i<3;i++){val(p,`A${63+i}`,`case ${i+1}`);input(p,`B${63+i}`,[16384,262144,1048576][i]);}
val(p,'C63','EP case 1');input(p,'D63',32);val(p,'C64','EP case 2');input(p,'D64',256);
note(p,67,'Pool align 默认：DS V4 为 C128 物理块对应的 128×128 raw tokens；其余为 128。实际启动若调整 state/cache group block，请填实际 LCM。');
note(p,68,'100% 命中时 scheduler 仍可能重算最后一个 token；pool worker 会拉完整末块。本页按实际块 payload 计量，不把命中长度简单减 1。');
title(p,70,'Cache component inputs · 参数表（BF16=2 B，FP32=4 B，INT8=1 B）');
header(p,71,['Model','Component','Layers','Elements / row','B / element','Extra B / row','Compress ratio','Block raw tokens','Window tokens','Row mode','TP mode','KV heads','State rows','Notes']);
let cr=72;const componentRows=new Map();
for(let i=0;i<models.length;i++){
  const rows=[];
  for(const component of models[i].components){
    const r=cr++;rows.push({r,component});val(p,`A${r}`,models[i].name);val(p,`B${r}`,component.name);
    for(const [col,v]of[['C',component.layers],['D',component.elements],['E',component.elementBytes],['F',component.extraBytes],['G',component.ratio],['I',component.window],['J',component.mode],['K',component.tpMode],['L',component.heads]])input(p,`${col}${r}`,v);
    f(p,`H${r}`,`$D$6*${component.blockFactor}`);p.getRange(`H${r}`).format.fill='#FFF2CC';
    if(component.name==='Conv state')f(p,`M${r}`,`${component.stateRows}+$H$6`);else val(p,`M${r}`,component.stateRows);
    p.getRange(`M${r}`).format.fill='#FFF2CC';val(p,`N${r}`,component.note);
  }
  componentRows.set(i,rows);
  const tr=54+i,validBlock=rows.map(({r})=>`MOD(D${tr},H${r})=0`).join(',');
  const validTP=['kimi','qwen38'].includes(models[i].profile)?`,B${tr}=C${tr}`:'';
  f(p,`F${tr}`,`IF(AND(B${tr}>0,D${tr}>0,MOD($D$63,B${tr})=0,MOD($D$64,B${tr})=0,${validBlock}${validTP}),"OK","CHECK inputs")`);
}
p.getRange(`B72:B${cr-1}`).format.columnWidth=26;p.getRange(`A72:N${cr-1}`).format={wrapText:true,rowHeight:39};
const detailTitle=cr+3,detailHeader=detailTitle+1;
title(p,detailTitle,'Transfer build · 逐组件字节计算（每个远端命中请求 / P rank）');
header(p,detailHeader,['Model','Input tokens','Pool-hit tokens','Component','Transferred rows','Bytes / row / rank','Layers','Bytes / rank','Kind']);
let dr=detailHeader+1;const builds=new Map();
for(let i=0;i<models.length;i++)for(let c=0;c<3;c++){
  const start=dr,cache=[],state=[],tr=54+i,sc=63+c;
  for(const {r:pr,component}of componentRows.get(i)){
    const r=dr++;val(p,`A${r}`,models[i].name);f(p,`B${r}`,`$B$${sc}`);f(p,`C${r}`,`INT(B${r}*$H$5/$D$${tr})*$D$${tr}`);f(p,`D${r}`,`B${pr}`);
    f(p,`E${r}`,`IF(C${r}=0,0,IF(J${pr}="endpoint",M${pr},IF(J${pr}="tail",MIN(C${r}/H${pr},ROUNDUP(MAX(0,I${pr}-1)/H${pr},0))*H${pr}/G${pr},C${r}/G${pr})))`);
    f(p,`F${r}`,`(D${pr}*E${pr}+F${pr})*IF(K${pr}="sharded",1/$B$${tr},IF(K${pr}="kv-heads",MAX(1,L${pr}/$B$${tr}),1))`);
    f(p,`G${r}`,`C${pr}`);f(p,`H${r}`,`E${r}*F${r}*G${r}`);val(p,`I${r}`,component.kind);
    (component.kind==='state'?state:cache).push(`H${r}`);
  }
  builds.set(`${i}:${c}`,{start,end:dr-1,cache,state});
}
const prefixRows=[];let mr=12;
for(let i=0;i<models.length;i++)for(let epcase=0;epcase<2;epcase++)for(let c=0;c<3;c++){
  const r=mr++,tr=54+i,b=builds.get(`${i}:${c}`);val(p,`A${r}`,models[i].name);
  f(p,`B${r}`,`$D$${63+epcase}`);f(p,`C${r}`,`$B$${tr}`);f(p,`D${r}`,`B${r}/C${r}`);f(p,`E${r}`,`B${b.start}`);f(p,`F${r}`,`C${b.start}`);
  f(p,`G${r}`,`IF($F$${tr}="OK",SUM(${b.cache.join(',')})/2^20,NA())`);
  f(p,`H${r}`,`IF($F$${tr}="OK",${b.state.length?`SUM(${b.state.join(',')})`:'0'}/2^20,NA())`);
  f(p,`I${r}`,`(G${r}+H${r})*C${r}`);f(p,`J${r}`,`(G${r}+H${r})*$D$5*(1-$L$5)/1024`);
  f(p,`K${r}`,`J${r}*C${r}`);f(p,`L${r}`,`J${r}*B${r}`);
  prefixRows.push({r,i,c,epcase});
}
p.getRange('G12:L47').setNumberFormat('#,##0.000000');p.getRange('A12:L47').format.rowHeight=26;
p.getRange(`D${detailHeader+1}:D${dr-1}`).format.columnWidth=25;p.getRange(`A${detailHeader+1}:I${dr-1}`).format={wrapText:true,rowHeight:35};
note(p,dr+2,'SWA / compressor tail: ceil((window−1)/block) 个末尾块；Mamba/KDA/GDN: 一个对齐端点状态；Full KV: 命中的全部历史。没有累计中间状态。');
note(p,dr+3,'V4.1 只计 40 个 SWA 与 4 个共享 KV / Indexer 源。私有 FP32 ring 与 Engram history 不在 prefix-hit payload 中；恢复路径需另外处理。');
note(p,dr+4,'本页算 host pool → device 的搬运字节；不包含 D→host 写入、allocator padding、协议/重试，也不将 byte budget 当成端到端可用性结论。');
note(p,dr+6,'Sources / reproducibility: analysis/HOST-TRANSFERS.md · analysis/host-transfer-sources.json · models/<model>/config.json');
note(p,dr+7,'https://github.com/GDzhu01/vllm-ascend-v41-private/tree/6bb7aeecf01b660ae14d9adddfc2f7035d3d1305/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store');
note(p,dr+8,'https://github.com/vllm-project/vllm/blob/c8438a3d40168ce1d9eade0dc15ccbe5d27adb68/vllm/v1/core/single_type_kv_cache_manager.py');

// Native input validation, with ordinary unprotected cells and visible grids.
for(const [s,range]of[[e,'D7'],[p,'H5'],[p,'L5']])s.dataValidations.add({range,rule:{type:'decimal',operator:'between',formula1:0,formula2:1}});
for(const [s,range]of[[e,'D6'],[e,'H6'],[p,'B54:C59']])s.dataValidations.add({range,rule:{type:'list',values:['1','2','4','8']}});
p.getRange('D6').dataValidation={rule:{type:'list',values:['32','64','128']}};
e.getRange('D8').dataValidation={rule:{type:'list',values:['0','1']}};
for(const [s,range]of[[e,'H5'],[e,'L5'],[p,'D5'],[p,'B63:B65']])s.dataValidations.add({range,rule:{type:'whole',operator:'between',formula1:1,formula2:10000000}});
p.getRange('H6').dataValidation={rule:{type:'whole',operator:'between',formula1:0,formula2:64}};
p.getRange('D63:D64').dataValidation={rule:{type:'list',values:['32','64','128','256']}};
for(const {r}of [...componentRows.values()].flat()){
  p.getRange(`J${r}`).dataValidation={rule:{type:'list',values:['full','tail','endpoint']}};
  p.getRange(`K${r}`).dataValidation={rule:{type:'list',values:['replicated','sharded','kv-heads']}};
}
const get=(s,c)=>s.getRange(c).values[0][0];
const near=(actual,expected,label)=>assert(Math.abs(actual-expected)<Math.max(1e-6,Math.abs(expected)*1e-11),`${label}: ${actual} != ${expected}`);
function verify(){
  for(const {r,group,k}of engramRows){
    const result=engramTransfer({layers:get(e,'B36'),ngram:get(e,'C36'),heads:get(e,'D36'),width:get(e,'E36')},{
      ep:get(p,`D${63+group}`),tp:k===1?get(e,'H6'):get(e,'D6'),tokensPerDP:get(e,`E${r}`),rounds:get(e,`F${r}`),
      ranksPerNode:get(e,'L6'),cpuVectorFraction:get(e,'D7'),vectorElementBytes:get(e,'H7'),metadataOnDevice:get(e,'D8')});
    for(const [col,key,scale]of[['G','vectorBytesPerRank',2**20],['H','auxiliaryBytesPerRank',2**20],['I','totalBytesPerRank',2**20],['J','totalBytesPerNode',2**30],['K','totalBytesPerEP',2**30],['L','leaderBytes',2**20]])near(get(e,`${col}${r}`)*scale,result[key],`Engram ${col}${r}`);
  }
  for(const {r,i,c,epcase}of prefixRows){
    const result=prefixTransfer(models[i],{inputTokens:get(p,`B${63+c}`),ep:get(p,`D${63+epcase}`),prefillTP:get(p,`B${54+i}`),decodeTP:get(p,`C${54+i}`),poolAlignment:get(p,`D${54+i}`),blockSize:get(p,'D6'),speculativeConvSlots:get(p,'H6'),requestsPerDP:get(p,'D5'),prefixHitFraction:get(p,'H5'),localRequestFraction:get(p,'L5')});
    for(const [col,key,scale]of[['G','cacheBytesPerRank',2**20],['H','stateBytesPerRank',2**20],['I','requestBytesAllTP',2**20],['J','rankBatchBytes',2**30],['K','dpBatchBytes',2**30],['L','epBatchBytes',2**30]])near(get(p,`${col}${r}`)*scale,result[key],`Prefix ${col}${r}`);
  }
}
wb.recalculate();verify();
// Boundary/driver recalculation checks; restore the delivered defaults afterwards.
for(const hit of [0,0.5,1]){val(p,'H5',hit);val(p,'B63',16383);val(p,'D5',3);val(p,'H6',7);val(e,'D7',0.5);wb.recalculate();verify();}
val(p,'L5',1);wb.recalculate();verify();
val(p,'H5',1);val(p,'B63',16384);val(p,'D5',16);val(p,'H6',0);val(p,'L5',0);val(e,'D7',1);
val(p,'D6',64);wb.recalculate();verify();val(p,'D6',128);wb.recalculate();verify();
for(const r of [57,58]){val(p,`B${r}`,4);val(p,`C${r}`,4);}wb.recalculate();verify();
for(const r of [57,58]){val(p,`B${r}`,8);val(p,`C${r}`,8);}wb.recalculate();verify();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:10},summary:'final formula error scan'});
console.log(errors.ndjson);
const file=path.join(out,'engram-h2d-prefix-rh2d.xlsx');await(await SpreadsheetFile.exportXlsx(wb)).save(file);
// Reopen the saved artifact before delivery, checking cached output and two sheets.
const reopened=await SpreadsheetFile.importXlsx(await FileBlob.load(file));
assert.equal(reopened.worksheets.items.length,2);
for(const [name,range]of[['Engram H2D','G12:L21'],['Prefix RH2D','G12:L47']]){
  assert.deepEqual(reopened.worksheets.getItem(name).getRange(range).values,wb.worksheets.getItem(name).getRange(range).values);
}
for(const [sheetName,range,name]of[['Engram H2D','A1:L26','engram'],['Engram H2D','A28:L38','engram-inputs'],['Prefix RH2D','A1:L24','prefix'],['Prefix RH2D','A30:L47','prefix-rest'],['Prefix RH2D','A50:L68','prefix-inputs'],['Prefix RH2D','A70:N78','prefix-components']]){
  const im=await reopened.render({sheetName,range,scale:1.25,format:'png'});
  await fs.writeFile(path.join(out,`${name}.png`),new Uint8Array(await im.arrayBuffer()));
}
await fs.writeFile(path.join(out,'results.json'),JSON.stringify(await hostTransferReport(),null,2)+'\n');
console.log(`Verified 10 Engram cases + 36 prefix cases, driver/boundary changes and saved XLSX: ${file}`);
