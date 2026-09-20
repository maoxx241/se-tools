import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {loadSpreadsheetRuntime} from '../lib/spreadsheet-runtime.mjs';
import {loadKvEpSpecs} from '../lib/kv-ep-specs.mjs';
import {MINUTE_CASES,minuteImpact} from '../lib/pd-minute.mjs';
import {decodeTpTraffic} from '../lib/communication-requirements.mjs';
import {addMinuteTp} from './minute-tp-workbook.mjs';
import {clarifyMinuteInputs} from './minute-input-language.mjs';
import {applyEditableSweepPresentation,cleanSweepText,cleanV41Labels} from './sweep-presentation.mjs';

const mode=process.argv[2]||'--inspect';
assert(['--inspect','--write','--inspect-sweep-style','--style-sweep','--add-tp','--inspect-v41-labels','--clean-v41-labels','--inspect-input-language','--clarify-input-language','--inspect-throughput-sweep','--add-throughput','--add-throughput-sweep'].includes(mode));
const root=path.resolve(process.argv[3]||'outputs/01a09481-77cf-7b72-a8db-64837639ac39/pd-minute');
const inputRoot=path.resolve(process.env.MINUTE_INPUT_ROOT||'.');
// Optional read-only formula extraction for Excel-resaved files whose shared
// formulas are not expanded by the importer. XLSX authoring remains here.
const formulaSnapshot=process.env.MINUTE_FORMULA_SNAPSHOT
  ? JSON.parse(await fs.readFile(process.env.MINUTE_FORMULA_SNAPSHOT,'utf8')) : null;
await fs.mkdir(root,{recursive:true});
const {FileBlob,SpreadsheetFile}=await loadSpreadsheetRuntime();
const specs=await loadKvEpSpecs();
const existing=JSON.parse(await fs.readFile('examples/pd-contention/model-results.json','utf8'));
let files=[{file:'examples/kv-ep-sweep/kv-ep32-ep256.xlsx',items:specs,style:'sweep'},
  ...specs.filter(x=>x.ep===32&&x.slug!=='deepseek-v4-10t').map(x=>({
    file:`models/${x.slug}/${x.slug}-analysis.xlsx`,items:specs.filter(y=>y.slug===x.slug),style:'model'}))];
if(mode.includes('sweep'))files=files.slice(0,1);
if(mode==='--add-tp')files=files.filter(x=>x.style==='sweep'||['kimi-k3','qwen3.8-2.4t-a95b'].includes(x.items[0].slug));
if(mode.endsWith('v41-labels'))files=files.filter(x=>x.style==='sweep'||x.items[0].slug==='deepseek-v4.1-flash');
const put=(s,c,v)=>{s.getRange(c).values=[[v]];};
const fx=(s,c,f)=>{s.getRange(c).formulas=[[`=${f}`]];};
const val=(s,c)=>s.getRange(c).values[0][0];
const close=(a,b)=>assert(Math.abs(a-b)<Math.max(1e-8,Math.abs(b)*1e-10),`${a} != ${b}`);
async function render(wb,sheet,range,file) {
  const blob=await wb.render({sheetName:sheet,range,scale:1.3,format:'png'});
  await fs.writeFile(path.join(root,file),new Uint8Array(await blob.arrayBuffer()));
}

function addMinuteThroughput(s,region,style) {
  const col=val(s,'P22')==='TP GB/s 输入'?'U':'Q',h=region.first-1;
  const base={name:style==='sweep'?'Carlito':'Arial',size:style==='sweep'?11:10,color:'#000000'};
  s.getRange(`${col}${h}:${col}${region.last}`).copyFrom(s.getRange(`N${h}:N${region.last}`),'all');
  put(s,`${col}${h}`,'Decode throughput\nloss (%)');
  s.getRange(`${col}${h}`).format={font:{...base,bold:true},fill:style==='sweep'?'#D9E1F2':'#D9E2F3',
    wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',borders:{preset:'all',style:'thin',color:'#A6A6A6'}};
  s.getRange(`${col}:${col}`).format.columnWidth=21;
  for(let r=region.first;r<=region.last;r++)fx(s,`${col}${r}`,`IF(ISNUMBER(L${r}),L${r}/(60000+L${r}),IF(L${r}="","",L${r}))`);
  s.getRange(`${col}${region.first}:${col}${region.last}`).setNumberFormat('[=0]0.000000%;[<0.00000001]"<0.000001%";0.000000%');
  put(s,`A${region.last+1}`,'Decode throughput loss = Total overhead / (60000 + Total overhead). Fixed batch size and tokens/step; Decode stays busy.');
  s.getRange(`A${region.last+1}`).format.font=base;
  return col;
}

function build(wb,items,style) {
  assert(!wb.worksheets.items.some(s=>s.name==='分钟场景'),'Minute sheet already exists; edit its inputs directly rather than reset them');
  const s=wb.worksheets.add('分钟场景'), dark=style==='sweep';
  const inputColor=dark?'#FFF0C2':'#FFF2CC';
  const base={name:'Arial',size:10,color:dark?'#202C3B':'#000000'};
  const head=(row,labels)=>{
    const r=s.getRangeByIndexes(row-1,0,1,labels.length);r.values=[labels];
    r.format={font:{...base,bold:true,color:dark?'#FFFFFF':'#000000'},fill:dark?'#253D5B':'#D9E2F3',
      horizontalAlignment:'center',verticalAlignment:'center',rowHeight:42,wrapText:true,
      ...(dark?{}:{borders:{preset:'all',style:'thin',color:'#A6A6A6'}})};
  };
  const input=(c,v,format='#,##0.00')=>{put(s,c,v);s.getRange(c).format.fill=inputColor;s.getRange(c).setNumberFormat(format);};
  const parameterFirst=23,parameterLast=parameterFirst+items.length-1;
  const outputHeader=parameterLast+5,first=outputHeader+1,last=first+items.length*12-1;
  const buildHeader=last+6,buildFirst=buildHeader+1,buildLast=buildFirst+items.length*12-1;
  const cacheHeader=buildLast+6;
  s.showGridLines=false;s.tabColor=dark?'#253D5B':'#D9E2F3';
  s.getRange(`A1:X${cacheHeader+240}`).format={font:base,rowHeight:dark?25:22,verticalAlignment:'center'};
  s.getRange('A:A').format.columnWidth=23;
  s.getRange('B:N').format.columnWidth=15;
  s.getRange('O:O').format.columnWidth=30;s.getRange('P:X').format.columnWidth=18;
  put(s,'A2','PD KV 传输：一分钟内的 Decode 通信暴露');s.getRange('A2').format.font={...base,size:14,bold:true};
  put(s,'A3','稳态相位平均估算。每层以等长 EP 通信段建模；未使用逐层 profiling，次数可为小数。');
  for(const [l,c,label,value] of [
    ['A5','B5','每 P DP 请求数',16],['E5','F5','P EP 组数',1],['I5','J5','D EP 组数',1],['M5','N5','KV 带宽 GB/s',4800],
    ['A7','B7','D tokens/DP/步',128],['E7','F7','产出 tokens/步',1],['I7','J7','窗口内带宽降幅',.1],['M7','N7','EP 带宽 GB/s',4800],
    ['A9','B9','每周期错峰批数',1],['E9','F9','通信共享比例',1],['I9','J9','关键路径暴露率',1],['M9','N9','KV block tokens',128],
    ['A11','B11','Conv 额外 slots',7]]) {
    put(s,l,label);input(c,value,['J7','F9','J9'].includes(c)?'0.0%':['N5','N7','F7'].includes(c)?'#,##0.00':'#,##0');
  }
  put(s,'D11','TTFT 暂作 P 批次间隔；步间隔＝TPOT×平均产出。每周期 1 批表示各 P DP 同步交接。');
  head(14,['输入场景','输入 tokens','TTFT / 周期 s','','TPOT 档位','TPOT ms']);
  MINUTE_CASES.forEach((c,i)=>{put(s,`A${15+i}`,['16K','256K','1M'][i]);input(`B${15+i}`,c.promptTokens,'#,##0');input(`C${15+i}`,c.ttftSeconds);});
  [5,10,20,30].forEach((v,i)=>{put(s,`E${15+i}`,i+1);input(`F${15+i}`,v);});
  put(s,'H15','黄色为输入；模型 / EP 带宽可在下表逐项覆盖。');
  put(s,'H16','时间增量针对基线 60 秒工作量；不表示实测吞吐或 P99。');
  put(s,'H17','TTFT 只用于到达节奏，本页估算 D 侧 TPOT；P 耗时仍见原表。');
  put(s,'H18','全 EP＝该步所有等效 EP 通信段都在 KV 窗口内；不代表整步持续时间都重叠。');
  head(22,['模型','EP','P TP','D TP','MoE 层数','Hidden','TopK','逻辑专家数','EP GB/s 输入','KV GB/s 输入','D tokens/DP/步','产出 tokens/步','共享比例','暴露率','配置条件']);
  items.forEach((m,i)=>{
    const r=parameterFirst+i;
    s.getRange(`A${r}:H${r}`).values=[[m.model,m.ep,Number(m.topology.prefillTP),Number(m.topology.decodeTP),m.facts.moeLayers,m.facts.H,m.facts.topK,m.facts.experts]];
    for(const [c,ref] of [['I','$N$7'],['J','$N$5'],['K','$B$7'],['L','$F$7'],['M','$F$9'],['N','$J$9']]) {
      fx(s,`${c}${r}`,ref);s.getRange(`${c}${r}`).format.fill=inputColor;
    }
    s.getRange(`M${r}:N${r}`).setNumberFormat('0.0%');
    put(s,`O${r}`,m.redundantExpertsRequired?'需 128 个冗余专家槽':'');
  });
  put(s,`A${outputHeader-2}`,'TPOT × 输入场景 × EP：稳态每分钟结果');s.getRange(`A${outputHeader-2}`).format.font={...base,bold:true};
  head(outputHeader,['模型','EP','输入 tokens','周期 s','TPOT ms','KV 窗口 次/min','KV 窗口 ms/次','重叠 FW 次/min','全 EP FW 次/min','部分 EP FW 次/min','重叠层调用 次/min','60s 工作增时 ms','平均 TPOT 增量 μs','TPOT 增幅 ppm','结果条件','最低 EP GB/s']);
  put(s,`A${buildHeader-3}`,'统计单位为一个 D DP 副本的 forward。EP 内各 Rank 并行，不能再把 forward 次数乘 EP。');
  put(s,`A${buildHeader-2}`,'交接请求总数、每 D DP 请求数、KV 占用和完整计算见下表。全 EP / 部分 EP 是基线窗口交集分类。');
  head(buildHeader,['输入有效','周期 ms','步间隔 ms','取 GiB/请求','P DP/组','D DP/组','全系统请求交接/min','每 D DP 请求/min','P Rank GiB/窗口','D Rank GiB/窗口','KV 窗口 ms','KV 间隔 ms','KV 负载率','EP GB/Rank/步','EP 带宽项 ms','每层 EP 段 ms','最低 EP GB/s','FW 重叠概率','全 EP 概率','每层平均增时 ms','基线 FW/min','60s 工作增时 ms','输入 / 调度状态','等效全通信 FW/min']);

  // KV rows retain model-specific SWA/compression/MLA/recurrent-state formulas.
  put(s,`A${cacheHeader-2}`,'KV 组件：输入长度变化时重新计算生成、保留和迁移字节');
  head(cacheHeader,['模型','输入 tokens','组件','层数','B/行/Rank','P 副本','D 副本','生成行','保留行','取/规划行','生成 GiB','保留 GiB','取/规划 GiB']);
  let cr=cacheHeader+1;
  const totals=new Map();
  for(const m of items.filter(x=>x.ep===32))for(let scenario=0;scenario<3;scenario++) {
    const begin=cr,S=`$B$${15+scenario}`,N=['glm53','deepseek_v41'].includes(m.profile)?S:`(${S}-1)`,B='$N$9';
    for(const p of m.cache.components) {
      const v41=m.profile==='deepseek_v41';let g,keep,pull;
      if(p.component.startsWith('SWA')) {
        const W=Number(p.retainedRows);g=N;keep=`MIN(${N},${W})`;
        pull=`MIN(ROUNDUP(${N}/${B},0),ROUNDUP(${W}/${B},0)+1)*${B}`;
      } else if(p.component.includes('circular ring')) {g=keep=`MIN(${N},32)`;pull='32';
      } else if(p.component.includes('compressor state')||p.component.includes('indexer state')) {
        const W=Number(p.retainedRows),bs=p.component.startsWith('C128')?`(${B}/4)`:`(${B}/16)`;
        g=keep=`MIN(${N},${W})`;pull=`MIN(ROUNDUP(${N}/${bs},0),ROUNDUP(${W}/${bs},0)+1)*${bs}`;
      } else if(p.component.startsWith('Conv')) {g=keep=`MIN(${N},${p.generatedRows})`;pull=`${p.generatedRows}+$B$11`;
      } else if(p.component.startsWith('Recurrent')) {g=keep=pull='1';
      } else {
        const ratio=p.component.startsWith('C128')?128:p.component.startsWith('C4')?4:p.component.startsWith('C2')?2:1;
        g=keep=`INT(${N}/${ratio})`;pull=v41?`ROUNDUP(INT(${N}/${ratio})/(${B}/${ratio}),0)*(${B}/${ratio})`:`ROUNDUP(${N}/(${ratio}*${B}),0)*${B}`;
      }
      put(s,`A${cr}`,m.model);fx(s,`B${cr}`,S);put(s,`C${cr}`,p.component);
      s.getRange(`D${cr}:G${cr}`).values=[[Number(p.layers),Number(p.rowBytes),Number(p.prefillCopies),Number(p.decodeCopies)]];
      const valid=`AND(ISNUMBER(${S}),${S}>=2,MOD(${S},1)=0,OR(${B}=32,${B}=64,${B}=128),ISNUMBER($B$11),$B$11>=0,MOD($B$11,1)=0)`;
      for(const [c,expression] of [['H',g],['I',keep],['J',pull],['K',`D${cr}*E${cr}*F${cr}*H${cr}/2^30`],['L',`D${cr}*E${cr}*F${cr}*I${cr}/2^30`],['M',`D${cr}*E${cr}*G${cr}*J${cr}/2^30`]])
        fx(s,`${c}${cr}`,`IF(${valid},${expression},"")`);
      cr++;
    }
    put(s,`A${cr}`,m.model);fx(s,`B${cr}`,S);put(s,`C${cr}`,'每请求合计');
    for(const c of ['K','L','M'])fx(s,`${c}${cr}`,`SUM(${c}${begin}:${c}${cr-1})`);
    s.getRange(`A${cr}:M${cr}`).format.font={...base,bold:true};
    totals.set(`${m.slug}:${scenario}`,cr);cr++;
  }
  const rows=[];let n=0;
  for(let i=0;i<items.length;i++)for(let scenario=0;scenario<3;scenario++)for(let t=0;t<4;t++) {
    const m=items[i],p=parameterFirst+i,r=first+n,b=buildFirst+n,k=totals.get(`${m.slug}:${scenario}`);n++;
    const ep=`B${p}`,tp=`C${p}`,dtp=`D${p}`,L=`E${p}`,H=`F${p}`,K=`G${p}`,E=`H${p}`,d='$J$7',a=`(1-${d})`;
    fx(s,`A${r}`,`A${p}`);fx(s,`B${r}`,ep);fx(s,`C${r}`,`$B$${15+scenario}`);fx(s,`D${r}`,`$C$${15+scenario}`);fx(s,`E${r}`,`$F$${15+t}`);
    const positives=['$B$5','$F$5','$J$5','$B$9',`I${p}`,`J${p}`,`K${p}`,`L${p}`,`D${r}`,`E${r}`];
    const valid=[...positives.map(x=>`AND(ISNUMBER(${x}),${x}>0)`),...['$B$5','$F$5','$J$5','$B$9',`K${p}`].map(x=>`MOD(${x},1)=0`),
      `ISNUMBER(C${r})`,`C${r}>=2`,`MOD(C${r},1)=0`,'OR($N$9=32,$N$9=64,$N$9=128)','ISNUMBER($B$11)','$B$11>=0','MOD($B$11,1)=0',
      `ISNUMBER(${d})`,`${d}>=0`,`${d}<1`,...['M','N'].map(c=>`AND(ISNUMBER(${c}${p}),${c}${p}>=0,${c}${p}<=1)`)];
    fx(s,`A${b}`,`AND(${valid.join(',')})`);
    const calc=(c,f)=>fx(s,`${c}${b}`,`IF(A${b},${f},"")`);
    calc('B',`D${r}*1000`);calc('C',`E${r}*L${p}`);calc('D',`M${k}`);
    calc('E',`${ep}/${tp}`);calc('F',`${ep}/${dtp}`);
    calc('G',`$F$5*E${b}*$B$5*60000/B${b}`);calc('H',`G${b}/($J$5*F${b})`);
    calc('I',`$B$5*D${b}/${tp}/$B$9`);calc('J',`I${b}*$F$5/$J$5`);
    calc('K',`MAX(I${b},J${b})*2^30/(J${p}*1e6)`);calc('L',`B${b}/$B$9`);calc('M',`K${b}/L${b}`);
    const record=`ROUNDUP((ROUNDUP(${H}*2/32,0)*32+32)/480,0)*512`;
    calc('N',`(${L}*ROUNDUP(K${p}/${dtp},0)*${K}*(${ep}-1)/${ep}*(${record}+${H}*2+32)+${L}*ROUNDUP(${E}/${ep},0)*(${ep}-1)*32)/1e9`);
    calc('O',`N${b}*1000/I${p}`);calc('P',`O${b}/${L}`);calc('Q',`N${b}*1000/C${b}`);calc('U',`60000/C${b}`);
    fx(s,`W${b}`,`IF(NOT(A${b}),"输入无效",IF(K${b}>=L${b},"KV 超载",IF(O${b}>C${b}+1e-9,"TPOT 不自洽",IF(OR(C${b}>L${b},P${b}>L${b}-K${b}),"需时序仿真","估算"))))`);
    const ok=`W${b}="估算"`,gap=`(C${b}-O${b})/${L}`,span=`((${L}-1)*C${b}/${L}+P${b})`;
    const probability=w=>`1-((${L}-1)*MAX(0,${gap}-(${w}))+MAX(0,L${b}-${span}-(${w})))/L${b}`;
    fx(s,`R${b}`,`IF(${ok},IF(OR(K${b}=0,O${b}=0),0,${probability(`K${b}`)}),"")`);
    fx(s,`S${b}`,`IF(${ok},IF(O${b}=0,0,MAX(0,1-(${probability(`L${b}-K${b}`)}))),"")`);
    fx(s,`T${b}`,`IF(${ok},IF(OR(P${b}=0,K${b}=0,${d}=0),0,IF(P${b}<=${a}*K${b},${d}*P${b}*K${b}/${a}-${d}^2*P${b}^2/(2*${a}^2),${d}*P${b}*K${b}+${d}^2*K${b}^2/2)/L${b}),"")`);
    fx(s,`V${b}`,`IF(${ok},U${b}*${L}*T${b}*M${p}*N${p},"")`);
    fx(s,`X${b}`,`IF(${ok},U${b}*M${b},"")`);
    fx(s,`F${r}`,`IF(A${b},60000/L${b},"")`);fx(s,`G${r}`,`K${b}`);
    for(const [c,f] of [['H',`U${b}*R${b}`],['I',`U${b}*S${b}`],['J',`U${b}*MAX(0,R${b}-S${b})`],
      ['K',`IF(OR(O${b}=0,K${b}=0),0,U${b}*${L}*MIN(1,(K${b}+P${b})/L${b}))`],['L',`V${b}`],['M',`E${r}*V${b}/60`],['N',`V${b}/60000*1e6`]])
      fx(s,`${c}${r}`,`IF(${ok},${f},"")`);
    fx(s,`O${r}`,`IF(${ok},"",W${b})`);fx(s,`P${r}`,`Q${b}`);
    rows.push({model:m,scenario,t,parameter:p,row:r,build:b,cache:k});
  }
  s.getRange(`C${first}:C${last}`).setNumberFormat('#,##0');s.getRange(`F${first}:K${last}`).setNumberFormat('#,##0.000');
  s.getRange(`L${first}:M${last}`).setNumberFormat('0.000000');s.getRange(`N${first}:N${last}`).setNumberFormat('0.000000');s.getRange(`P${first}:P${last}`).setNumberFormat('0.000');
  s.getRange(`B${buildFirst}:X${buildLast}`).setNumberFormat('#,##0.000000');s.getRange(`M${buildFirst}:M${buildLast}`).setNumberFormat('0.000000%');
  s.getRange(`K${cacheHeader+1}:M${cr}`).setNumberFormat('#,##0.000000');
  s.getRange(`C${cacheHeader+1}:C${cr}`).format.wrapText=true;s.getRange(`A${cacheHeader+1}:M${cr}`).format.rowHeight=45;
  s.getRange('N9').dataValidation={rule:{type:'list',values:['32','64','128']}};
  for(const text of ['输入无效','KV 超载','TPOT 不自洽','需时序仿真'])s.getRange(`O${first}:O${last}`).conditionalFormats.add('containsText',{text,format:{fill:'#FDE9E7',font:{color:'#A32020'}}});
  put(s,`A${cr+3}`,'来源：库内 config.json、analysis/KV-EP-SPECS.md、analysis/PD-MINUTE.md。');
  put(s,`A${cr+4}`,'MC2：A3 FullMesh BF16；均衡路由。未计 TP / Attention / Engram、通信启动和真实排队，未复现 NPU 实测。');
  if(style==='sweep')applyEditableSweepPresentation(wb,{sheetNames:['分钟场景']});
  const region={s,rows,first,last,parameterFirst,parameterLast,buildFirst,buildLast,cacheHeader,cacheLast:cr};
  addMinuteTp(s,region,items);
  addMinuteThroughput(s,region,style);
  const used=s.getUsedRange(),values=used.values,formulas=used.formulas;
  for(let i=0;i<values.length;i++)for(let j=0;j<values[i].length;j++) {
    const f=formulas[i]?.[j],v=f||values[i][j],next=cleanV41Labels(v);
    if(next!==v) {if(f)used.getCell(i,j).formulas=[[next]];else used.getCell(i,j).values=[[next]];}
  }
  return region;
}

async function verify(wb,region) {
  const {s,rows}=region;
  for(const x of rows) {
    const m=x.model,p=x.parameter,r=x.row;
    const kv=await loadKvEpSpecs({promptTokens:val(s,`C${r}`),requestsPerDP:val(s,'B5'),
      decodeTokensPerDP:val(s,`K${p}`),blockSize:val(s,'N9'),speculativeSlots:val(s,'B11')});
    const source=kv.find(v=>v.slug===m.slug&&v.ep===m.ep);
    close(val(s,`M${x.cache}`)*2**30,Number(source.cache.pullBytes??source.cache.plannedPullBytes));
    close(val(s,`K${x.cache}`)*2**30,Number(source.cache.generatedBytes));
    close(val(s,`L${x.cache}`)*2**30,Number(source.cache.retainedBytes));
    const expect=minuteImpact({pullBytesPerRequest:Number(source.cache.pullBytes??source.cache.plannedPullBytes),
      ep:m.ep,prefillTP:Number(m.topology.prefillTP),decodeTP:Number(m.topology.decodeTP),
      decodeBytesPerRank:Number(source.decodeEP.modeledRemoteWriteBytes)/m.ep,layers:m.facts.moeLayers,
      decodeTpBytesPerRank:decodeTpTraffic(m.facts,val(s,`K${p}`),m.topology.decodeTP,{sharedExpertTpEnabled:m.profile==='kimi'}).totalBytesPerRank,
      tpBandwidthGBps:val(s,`P${p}`),tpSharedFraction:val(s,`Q${p}`),tpExposedFraction:val(s,`R${p}`),
      ttftSeconds:val(s,`D${r}`),tpotMs:val(s,`E${r}`),requestsPerDP:val(s,'B5'),prefillGroups:val(s,'F5'),decodeGroups:val(s,'J5'),
      kvBandwidthGBps:val(s,`J${p}`),decodeBandwidthGBps:val(s,`I${p}`),decodeTokensPerDP:val(s,`K${p}`),outputTokensPerStep:val(s,`L${p}`),
      burstsPerCycle:val(s,'B9'),bandwidthLoss:val(s,'J7'),sharedFraction:val(s,`M${p}`),exposedFraction:val(s,`N${p}`)});
    assert.equal(expect.status,'ok');
    for(const [c,key] of [['F','kvBurstsPerMinute'],['G','windowMs'],['H','overlappingStepsPerMinute'],['I','allCommunicationInsideStepsPerMinute'],
      ['J','partialStepsPerMinute'],['K','overlappingLayerCallsPerMinute'],['L','extraMsForBaselineMinute'],['M','meanTpotIncreaseMs'],['N','tpotIncreaseFraction'],['P','minimumDecodeBandwidthGBps']])close(val(s,`${c}${r}`),expect[key]*(c==='M'?1000:c==='N'?1e6:1));
    close(val(s,`R${r}`),expect.tpCommMs);close(val(s,`S${r}`),expect.extraEpMsForBaselineMinute);close(val(s,`T${r}`),expect.extraTpMsForBaselineMinute);
  }
}

const results=[];
if(mode==='--inspect-sweep-style'&&process.argv[4]) {
  const reference=await SpreadsheetFile.importXlsx(await FileBlob.load(process.argv[4]));
  await render(reference,reference.worksheets.getItemAt(0).name,'A1:L14','reference-kimi.png');
}
for(const entry of files) {
  const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(inputRoot,entry.file)));
  if(formulaSnapshot?.[entry.file]) {
    for(const [sheet,runs] of Object.entries(formulaSnapshot[entry.file]))for(const [range,formulas] of runs)
      wb.worksheets.getItem(sheet).getRange(range).formulas=[formulas];
    wb.recalculate();
  }
  const slug=entry.items.length>2?'ep32-ep256':entry.items[0].slug;
  if(mode.includes('throughput')) {
    const region=JSON.parse(await fs.readFile('examples/pd-contention/minute-workbook-regions.json','utf8')).find(x=>x.file===entry.file);
    const s=wb.worksheets.getItem('分钟场景');
    if(mode.startsWith('--inspect')) {
      await render(wb,s.name,`L${region.first-1}:U${region.first+5}`,`${slug}-before.png`);continue;
    }
    const before=wb.worksheets.items.map(sheet=>({sheet,range:sheet.getUsedRange(),values:sheet.getUsedRange().values,formulas:sheet.getUsedRange().formulas}));
    const col=addMinuteThroughput(s,region,entry.style),column=col==='U'?20:16;
    wb.recalculate();
    for(const old of before) {
      const now=old.range.values,formulas=old.range.formulas;
      old.values.forEach((row,i)=>row.forEach((v,j)=>{
        if(old.sheet.name===s.name&&((j===column&&i>=region.first-2&&i<region.last)||(j===0&&i===region.last)))return;
        assert.equal(formulas[i]?.[j]??'',old.formulas[i]?.[j]??'');
        if(typeof v==='number')close(now[i][j],v);else assert.equal(now[i][j]??'',v??'');
      }));
    }
    const check=()=>{for(let r=region.first;r<=region.last;r++) {
      const extra=val(s,`L${r}`),loss=val(s,`${col}${r}`);
      if(extra==='')assert.equal(loss,'');
      else close(loss,1-val(s,`E${r}`)/(val(s,`E${r}`)+val(s,`M${r}`)/1000));
    }};
    check();
    const saved=val(s,'J7');put(s,'J7',0);wb.recalculate();check();assert.equal(val(s,`${col}${region.first}`),0);
    put(s,'J7',saved);const bandwidth=val(s,'N7');put(s,'N7',0);wb.recalculate();check();assert.equal(val(s,`${col}${region.first}`),'');
    put(s,'N7',bandwidth);wb.recalculate();check();
    const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10}});
    assert(!/"kind":"match"/.test(errors.ndjson),errors.ndjson);
    await render(wb,s.name,`L${region.first-1}:${col}${region.first+5}`,`${slug}-throughput.png`);
    const dest=path.join(root,entry.file);await fs.mkdir(path.dirname(dest),{recursive:true});await(await SpreadsheetFile.exportXlsx(wb)).save(dest);
    console.log(`Added throughput loss: ${entry.file}; ${region.cases.length} cases; existing formulas and inputs preserved`);
    continue;
  }
  if(mode.endsWith('input-language')) {
    const region=JSON.parse(await fs.readFile('examples/pd-contention/minute-workbook-regions.json','utf8')).find(x=>x.file===entry.file);
    if(mode==='--clarify-input-language') {
      const before=wb.worksheets.items.map(s=>({s,range:s.getUsedRange(),values:s.getUsedRange().values,formulas:s.getUsedRange().formulas}));
      const edits=clarifyMinuteInputs(wb.worksheets.getItem('分钟场景'),region);
      wb.recalculate();
      for(const old of before) {
        const now=old.range.values;
        assert.deepEqual(old.range.formulas,old.formulas);
        old.values.forEach((row,i)=>row.forEach((v,j)=>{
          // Only literal labels in the authorized input panel may change.
          let n=j+1,col='';while(n){n--;col=String.fromCharCode(65+n%26)+col;n=Math.floor(n/26);}
          const label=old.s.name==='分钟场景'?edits.get(`${col}${i+1}`):undefined;
          if(label!==undefined)assert.equal(now[i][j]??'',label);
          else if(typeof v==='number')close(now[i][j],v);
          else assert.equal(now[i][j]??'',v??'');
        }));
      }
      const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:10}});
      assert(!/"kind":"match"/.test(errors.ndjson),errors.ndjson);
      const dest=path.join(root,entry.file);await fs.mkdir(path.dirname(dest),{recursive:true});await(await SpreadsheetFile.exportXlsx(wb)).save(dest);
      console.log(`Clarified inputs; all formulas and numeric results unchanged: ${entry.file}`);
    }
    const prefix=mode==='--clarify-input-language'?'after':'before';
    await render(wb,'分钟场景','A5:N13',`${prefix}-${slug}-inputs.png`);
    await render(wb,'分钟场景','H15:U22',`${prefix}-${slug}-help.png`);
    await render(wb,'分钟场景',`L${region.first-1}:T${region.first+3}`,`${prefix}-${slug}-totals.png`);
    continue;
  }
  if(mode.endsWith('v41-labels')) {
    if(mode==='--clean-v41-labels') {
      const before=wb.worksheets.items.map(s=>({s,range:s.getUsedRange(),values:s.getUsedRange().values,formulas:s.getUsedRange().formulas}));
      let count=0;
      for(const old of before)old.values.forEach((row,i)=>row.forEach((v,j)=>{
        const f=old.formulas[i]?.[j],next=cleanV41Labels(f||v);
        if(next!==(f||v)) {if(f)old.range.getCell(i,j).formulas=[[next]];else old.range.getCell(i,j).values=[[next]];count++;}
      }));
      wb.recalculate();
      for(const old of before) {
        const values=old.range.values,formulas=old.range.formulas;
        old.values.forEach((row,i)=>row.forEach((v,j)=>{
          assert.equal(formulas[i]?.[j]??'',cleanV41Labels(old.formulas[i]?.[j]??''));
          if(typeof v==='number')close(values[i][j],v);
          else assert.equal(values[i][j]??'',cleanV41Labels(v??''),`${old.s.name} row ${i+1} col ${j+1}`);
        }));
      }
      const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:10}});
      assert(!/"kind":"match"/.test(errors.ndjson),errors.ndjson);
      const dest=path.join(root,entry.file);await fs.mkdir(path.dirname(dest),{recursive:true});await(await SpreadsheetFile.exportXlsx(wb)).save(dest);
      console.log(`Removed ${count} display labels; formulas and numeric results preserved: ${entry.file}`);
    }
    const prefix=mode==='--clean-v41-labels'?'after':'before';
    if(entry.style==='sweep') {
      await render(wb,'规格汇总','G23:M35',`${prefix}-${slug}-summary.png`);
      await render(wb,'KV明细','G39:M47',`${prefix}-${slug}-kv.png`);
      await render(wb,'分钟场景','L159:T167',`${prefix}-${slug}-minute.png`);
    } else {
      await render(wb,'分钟场景','L22:P35',`${prefix}-${slug}-minute.png`);
      await render(wb,'Decode','A141:J143',`${prefix}-${slug}-note.png`);
    }
    continue;
  }
  if(mode==='--add-tp') {
    const original=wb.worksheets.items.filter(s=>s.name!=='分钟场景').map(s=>({s,range:s.getUsedRange(),values:s.getUsedRange().values,formulas:s.getUsedRange().formulas}));
    const meta=JSON.parse(await fs.readFile('examples/pd-contention/minute-workbook-regions.json','utf8')).find(x=>x.file===entry.file);
    const s=wb.worksheets.getItem('分钟场景');
    await render(wb,s.name,'A2:P18',`${slug}-before.png`);
    const region={...meta,s,rows:meta.cases.map((x,i)=>({model:entry.items.find(m=>m.slug===x.slug&&m.ep===x.ep),
      parameter:meta.parameterFirst+Math.floor(i/12),row:x.row,build:meta.buildFirst+i,
      cache:meta.cacheHeader+1+entry.items.filter(m=>m.ep===32).slice(0,entry.items.filter(m=>m.ep===32).findIndex(m=>m.slug===x.slug)).reduce((n,m)=>n+3*(m.cache.components.length+1),0)
        +x.scenario*(entry.items.find(m=>m.slug===x.slug).cache.components.length+1)+entry.items.find(m=>m.slug===x.slug).cache.components.length}))};
    addMinuteTp(s,region,entry.items);wb.recalculate();await verify(wb,region);
    const sample=region.rows.find(x=>Number(x.model.topology.decodeTP)>1),r=sample.row;
    const controls=['F11','J11','N11','N5','N7','B7'],saved=controls.map(c=>[c,val(s,c)]);
    const base=val(s,`L${r}`),ep=val(s,`S${r}`),tp=val(s,`T${r}`);
    put(s,'J11',0);wb.recalculate();await verify(wb,region);close(val(s,`L${r}`),ep);close(val(s,`T${r}`),0);
    put(s,'J11',1);put(s,'F11',2400);wb.recalculate();await verify(wb,region);assert(val(s,`T${r}`)>tp);close(val(s,`S${r}`),ep);
    put(s,'B7',129);wb.recalculate();await verify(wb,region);
    put(s,'F11',0);wb.recalculate();assert.equal(val(s,`O${r}`),'输入无效');assert.equal(val(s,`L${r}`),'');
    put(s,'F11',.001);wb.recalculate();assert.equal(val(s,`O${r}`),'TPOT 不自洽');
    for(const [c,v] of saved)put(s,c,v);wb.recalculate();await verify(wb,region);close(val(s,`L${r}`),base);
    for(const old of original) {
      assert.deepEqual(old.range.formulas,old.formulas);
      const now=old.range.values;
      old.values.forEach((row,i)=>row.forEach((v,j)=>typeof v==='number'?close(now[i][j],v):assert.equal(now[i][j]??'',v??'')));
    }
    const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:10}});
    assert(!/"kind":"match"/.test(errors.ndjson),errors.ndjson);
    console.log((await wb.inspect({kind:'table',range:`分钟场景!L${r}:T${r+1}`,include:'values,formulas',tableMaxRows:2,tableMaxCols:9})).ndjson);
    await render(wb,s.name,'A2:P18',`${slug}-inputs.png`);
    await render(wb,s.name,`L${region.first-1}:T${region.first+7}`,`${slug}-tp-results.png`);
    await render(wb,s.name,`P22:U${region.parameterLast}`,`${slug}-tp-controls.png`);
    const dest=path.join(root,entry.file);await fs.mkdir(path.dirname(dest),{recursive:true});await(await SpreadsheetFile.exportXlsx(wb)).save(dest);
    console.log(`Verified TP extension: ${entry.file}; ${region.rows.length} cases; other sheets unchanged`);
    continue;
  }
  if(mode==='--inspect-sweep-style'||mode==='--style-sweep') {
    if(mode==='--style-sweep') {
      const before=wb.worksheets.items.map(s=>({s,range:s.getUsedRange(),values:s.getUsedRange().values,formulas:s.getUsedRange().formulas}));
      console.log(JSON.stringify(applyEditableSweepPresentation(wb)));
      wb.recalculate();
      for(const old of before) {
        const current=old.range.values,afterFormulas=old.range.formulas;
        old.values.forEach((row,i)=>row.forEach((v,j)=>{
          assert.equal(afterFormulas[i]?.[j]??'',cleanSweepText(old.formulas[i]?.[j]??''));
          if(typeof v==='number')close(current[i][j],v);
          else if(typeof v==='boolean')assert.equal(current[i][j],v);
          else assert.equal(current[i][j]??'',cleanSweepText(v??''));
        }));
      }
      const s=wb.worksheets.getItem('分钟场景'),bandwidth=val(s,'N7'),base=val(s,'L41');
      put(s,'N7',bandwidth/2);wb.recalculate();assert(val(s,'L41')>base);
      put(s,'N7',0);wb.recalculate();assert.equal(val(s,'O41'),'输入无效');assert.equal(val(s,'L41'),'');
      put(s,'N7',bandwidth);wb.recalculate();close(val(s,'L41'),base);
      const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:10}});
      assert(!/"kind":"match"/.test(errors.ndjson),errors.ndjson);
      const dest=path.join(root,entry.file);await fs.mkdir(path.dirname(dest),{recursive:true});await(await SpreadsheetFile.exportXlsx(wb)).save(dest);
    }
    for(const [sheet,range,label] of [['规格汇总','A1:P18','summary'],['KV明细','A1:N16','kv'],['EP通信','A1:N21','ep'],
      ['分钟场景','A2:P18','minute-inputs'],['分钟场景','A22:P45','minute-results'],['分钟场景','Q180:Z198','minute-blank-grid']])
      await render(wb,sheet,range,`${mode==='--style-sweep'?'after':'before'}-${label}.png`);
    continue;
  }
  if(mode==='--inspect') {
    const row=entry.style==='sweep'?51:existing.find(x=>x.slug===slug&&x.phase==='Decode').region.start;
    await render(wb,entry.style==='sweep'?'规格汇总':'Decode',`A${row}:L${row+10}`,`${slug}-before.png`);continue;
  }
  const original=wb.worksheets.items.map(s=>({s,range:s.getUsedRange(),values:s.getUsedRange().values,formulas:s.getUsedRange().formulas}));
  const region=build(wb,entry.items,entry.style);wb.recalculate();await verify(wb,region);
  // Mutations independently exercise all rows, late scenarios and separate bandwidths.
  const changes=[['N5',2400],['N7',2400],['B5',8],['F5',2],['J5',3],['B9',4],['F7',2],['B17',786432],['C17',80],['F18',40]];
  for(const [c,v] of changes)put(region.s,c,v);wb.recalculate();await verify(wb,region);
  const defaults=[['N5',4800],['N7',4800],['B5',16],['F5',1],['J5',1],['B9',1],['F7',1],['B17',1048576],['C17',60],['F18',30]];
  for(const [c,v] of defaults)put(region.s,c,v);
  put(region.s,'N5',0);wb.recalculate();assert.equal(val(region.s,`O${region.first}`),'输入无效');assert.equal(val(region.s,`L${region.first}`),'');
  put(region.s,'N5',.001);wb.recalculate();assert.equal(val(region.s,`O${region.first}`),'KV 超载');
  put(region.s,'N5',4800);put(region.s,'N7',1);wb.recalculate();assert.equal(val(region.s,`O${region.first}`),'TPOT 不自洽');
  put(region.s,'N7',4800);wb.recalculate();await verify(wb,region);
  for(const old of original) {
    assert.deepEqual(old.range.formulas,old.formulas);
    const current=old.range.values;
    old.values.forEach((row,i)=>row.forEach((v,j)=>{
      if(typeof v==='number')close(current[i][j],v);
      else assert.equal(current[i][j]??'',v??'',`${old.s.name} ${i},${j}`);
    }));
  }
  const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10}});
  assert(!/"kind":"match"/.test(errors.ndjson),errors.ndjson);
  await render(wb,'分钟场景','A2:O18',`${slug}-inputs.png`);
  await render(wb,'分钟场景',`A${region.first-3}:P${region.first+11}`,`${slug}-cases.png`);
  await render(wb,'分钟场景',`A${region.cacheHeader-2}:M${region.cacheHeader+6}`,`${slug}-cache.png`);
  const dest=path.join(root,entry.file);await fs.mkdir(path.dirname(dest),{recursive:true});await(await SpreadsheetFile.exportXlsx(wb)).save(dest);
  results.push({file:entry.file,sheet:'分钟场景',first:region.first,last:region.last,parameterFirst:region.parameterFirst,parameterLast:region.parameterLast,
    buildFirst:region.buildFirst,buildLast:region.buildLast,cacheHeader:region.cacheHeader,cacheLast:region.cacheLast,
    cases:region.rows.map(x=>({slug:x.model.slug,ep:x.model.ep,scenario:x.scenario,tpotIndex:x.t,row:x.row}))});
  console.log(`Verified and exported ${entry.file}: ${region.rows.length} cases; original sheets unchanged`);
}
if(mode==='--write')await fs.writeFile(path.join(root,'workbook-regions.json'),JSON.stringify(results,null,2)+'\n');
