import assert from 'node:assert/strict';
import { contentionImpact } from '../lib/pd-contention.mjs';

const put=(s,c,v)=>{s.getRange(c).values=[[v]];};
const fx=(s,c,f)=>{s.getRange(c).formulas=[[`=${f}`]];};
const get=(s,c)=>s.getRange(c).values[0][0];
const header=(s,row,labels)=>{
  const r=s.getRangeByIndexes(row-1,0,1,labels.length);r.values=[labels];
  r.format={fill:'#D9E2F3',font:{bold:true,name:'Arial',size:10},rowHeight:38,wrapText:true,verticalAlignment:'center'};
};
const percent=(s,c)=>{s.getRange(c).setNumberFormat('0.0%');s.getRange(c).format.fill='#FFF2CC';};

export function findCommunication(sheet) {
  const values=sheet.getRange('A1:P260').values;
  const h=values.findIndex(r=>r[0]==='并行策略' && r[3]==='Collective')+1;
  assert(h>0, 'Missing communication table');
  let end=h+1;
  while(values[end-1]?.[3]) end++;
  assert(end>h+1,'Empty communication table');
  return {first:h+1,last:end-1,total:end,rows:values.slice(h,end-1)};
}

// Append below the existing table. All original weights, inputs and formulas stay intact.
export function addModelContention(sheet, phase, facts, profile) {
  const comm=findCommunication(sheet), start=Math.max(comm.total+3,70);
  const input=start+3, output=start+5, h=start+8, first=h+1;
  const loss=`$B$${input}`, coverage=`$D$${input}`, bw=`$H$${input}`, baseline=`$F$${input}`;
  const data=comm.rows.map((v,i)=>({source:comm.first+i,group:v[0],module:v[1],event:v[2],bytes:`K${comm.first+i}*2^20`}));
  const dispatch=data.find(x=>String(x.group).startsWith('EP') && /Dispatch/.test(x.event));
  const combine=data.find(x=>String(x.group).startsWith('EP') && /Combine/.test(x.event));
  if(dispatch && combine) {
    const d=dispatch.source,c=combine.source,p=`G${d}`,layers=`H${d}`;
    if(phase==='Decode') {
      const record=`ROUNDUP((ROUNDUP(${facts.H}*F${d}/32,0)*32+32)/480,0)*512`;
      dispatch.bytes=`IF(F${d}=2,E${d}/${facts.H}*H${d}*(${record}),NA())`;
      combine.bytes=`IF(F${c}=2,K${c}*2^20+E${c}/${facts.H}*H${c}*32,NA())`;
      dispatch.event='MC2 FullMesh Dispatch';combine.event='MC2 Combine + flags';
    }
    const count=phase==='Prefill'?`${layers}*ROUNDUP(${facts.experts}/${p},0)*${p}*4*(${p}-1)`:
      `${layers}*ROUNDUP(${facts.experts}/${p},0)*(${p}-1)*32`;
    data.push({group:'EP',module:'MoE 计数',event:phase==='Prefill'?'直方图 Ring AG':'MC2 count 写入',
      bytes:`IF(E${d}=0,0,${count})`});
  }
  const last=first+data.length-1,total=last+1;
  sheet.getRange(`A${start}:L${total}`).format={font:{name:'Arial',size:10,color:'#000000'},verticalAlignment:'center',rowHeight:23};
  put(sheet,`A${start}`,'PD 带宽争用耗时评估');sheet.getRange(`A${start}`).format.font={bold:true,size:14};
  put(sheet,`A${start+1}`,`${phase} 当前表参数；100 GB/s 为归一化示例。所列事件的带宽项，不等于完整 forward 耗时。`);
  header(sheet,input-1,['带宽降幅','输入','争用覆盖率','输入','基线 forward ms','实测输入','默认带宽 GB/s','输入']);
  put(sheet,`B${input}`,.1);put(sheet,`D${input}`,1);put(sheet,`H${input}`,100);
  percent(sheet,`B${input}`);percent(sheet,`D${input}`);sheet.getRange(`F${input}`).format.fill='#FFF2CC';sheet.getRange(`H${input}`).format.fill='#FFF2CC';
  header(sheet,output-1,['通信基线 ms','结果','通信争用后 ms','结果','forward 增量 ms','结果','forward 新耗时 ms','结果','forward 增幅','结果','输入状态','状态']);
  fx(sheet,`B${output}`,`SUM(G${first}:G${last})`);fx(sheet,`D${output}`,`SUM(H${first}:H${last})`);
  fx(sheet,`F${output}`,`SUM(K${first}:K${last})`);
  const valid=`AND(ISNUMBER(${baseline}),${baseline}>0,${baseline}>=SUMPRODUCT(G${first}:G${last},J${first}:J${last}))`;
  fx(sheet,`H${output}`,`IF(${valid},${baseline}+F${output},"")`);
  fx(sheet,`J${output}`,`IF(${valid},F${output}/${baseline},"")`);
  fx(sheet,`L${output}`,`IF(${baseline}="","待填基线",IF(${valid},"已填基线","基线/暴露不一致"))`);
  sheet.getRange(`A${output}:L${output}`).setNumberFormat('0.000');sheet.getRange(`J${output}`).setNumberFormat('0.00%');
  put(sheet,`A${start+6}`,'共享比例=0 表示独立链路；暴露系数=0 表示增量完全隐藏。默认均为 1，降幅只作用于带宽项。');
  put(sheet,`A${start+7}`,profile==='deepseek_v41'?'V4.1 为 PD 条件评估；旧 KV 规划与新 main 的端到端迁移可用性未在本次验证。':
    'P: AllToAllV + 4 B 计数 Ring；D: A3 BF16 MC2 FullMesh。每事件带宽按端点发送量定义。');
  header(sheet,h,['并行域','模块','通信事件','GB/Rank/Step','有效带宽 GB/s','共享比例','基线带宽项 ms','争用后带宽项 ms','通信增量 ms','暴露系数','forward 增量 ms','公式口径']);
  data.forEach((e,i)=>{
    const r=first+i;
    sheet.getRange(`A${r}:C${r}`).values=[[e.group,e.module,e.event]];
    fx(sheet,`D${r}`,`(${e.bytes})/1e9`);fx(sheet,`E${r}`,bw);put(sheet,`F${r}`,1);put(sheet,`J${r}`,1);
    const ok=`AND(ISNUMBER(E${r}),E${r}>0,ISNUMBER(${loss}),${loss}>=0,${loss}<1,ISNUMBER(${coverage}),${coverage}>=0,${coverage}<=1,ISNUMBER(F${r}),F${r}>=0,F${r}<=1,ISNUMBER(J${r}),J${r}>=0,J${r}<=1)`;
    fx(sheet,`G${r}`,`IF(${ok},D${r}/E${r}*1000,NA())`);
    fx(sheet,`I${r}`,`G${r}*F${r}*${coverage}*${loss}/(1-${loss})`);
    fx(sheet,`H${r}`,`G${r}+I${r}`);fx(sheet,`K${r}`,`I${r}*J${r}`);
    put(sheet,`L${r}`,e.source?`原通信行 ${e.source}`:'额外控制量');
  });
  for(const c of ['D','E','G','H','I','K'])sheet.getRange(`${c}${first}:${c}${total}`).setNumberFormat('0.000000');
  for(const c of ['E','F','J'])sheet.getRange(`${c}${first}:${c}${last}`).format.fill='#FFF2CC';
  for(const c of ['F','J'])sheet.getRange(`${c}${first}:${c}${last}`).setNumberFormat('0%');
  put(sheet,`A${total}`,'所列事件串行相加');for(const c of ['D','G','H','I','K'])fx(sheet,`${c}${total}`,`SUM(${c}${first}:${c}${last})`);
  sheet.getRange(`A${total}:L${total}`).format.fill='#E8EEF5';
  sheet.getRange(`L${output}`).conditionalFormats.add('containsText',{text:'不一致',format:{fill:'#FDE9E7',font:{color:'#A32020'}}});
  return {start,input,output,first,last,total,phase,comm};
}

export function verifyModelContention(wb,sheet,region) {
  const {input,output,first,last}=region;
  const expected=()=>{
    let base=0,changed=0,delta=0;
    for(let r=first;r<=last;r++) {
      const x=contentionImpact(get(sheet,`D${r}`)*1e9,{bandwidthGBps:get(sheet,`E${r}`),bandwidthLoss:get(sheet,`B${input}`),contentionCoverage:get(sheet,`D${input}`),sharedFraction:get(sheet,`F${r}`),exposedFraction:get(sheet,`J${r}`)});
      base+=x.baselineBandwidthMs;changed+=x.contendedBandwidthMs;delta+=x.deltaForwardMs;
    }
    for(const [c,n] of [['B',base],['D',changed],['F',delta]])assert(Math.abs(get(sheet,`${c}${output}`)-n)<1e-7);
    return {baselineBandwidthMs:base,contendedBandwidthMs:changed,deltaForwardMs:delta};
  };
  wb.recalculate();const baseline=expected();assert.equal(get(sheet,`H${output}`),'');
  put(sheet,`B${input}`,0);wb.recalculate();expected();assert.equal(get(sheet,`F${output}`),0);
  put(sheet,`B${input}`,.1);put(sheet,`D${input}`,.5);put(sheet,`H${input}`,50);
  put(sheet,`F${input}`,baseline.baselineBandwidthMs*3+1);wb.recalculate();expected();assert(get(sheet,`J${output}`)>=0);
  put(sheet,`D${input}`,1);put(sheet,`H${input}`,100);put(sheet,`F${input}`,null);wb.recalculate();expected();
  return baseline;
}

export function addSweepContention(sheet,cases) {
  sheet.getRange(`A51:O${61+cases.length}`).format={font:{name:'Arial',size:10,color:'#000000'},verticalAlignment:'center',rowHeight:25};
  sheet.getRange('N:N').format.columnWidth=18;sheet.getRange('O:O').format.columnWidth=36;
  put(sheet,'A51','PD 带宽争用：EP32 / EP256');sheet.getRange('A51').format.font={bold:true,size:14};
  put(sheet,'A52','只计 MoE EP 带宽项，单位 ms/Step/Rank。100 GB/s 是归一化示例；全组字节先除 EP，不能直接除单卡带宽。');
  header(sheet,54,['P GB/s','输入','D GB/s','输入','带宽降幅','输入','争用覆盖率','输入','暴露系数','输入']);
  put(sheet,'B55',100);put(sheet,'D55',100);put(sheet,'F55',.1);put(sheet,'H55',1);put(sheet,'J55',1);
  for(const c of ['B55','D55','F55','H55','J55'])sheet.getRange(c).format.fill='#FFF2CC';
  for(const c of ['F55','H55','J55'])sheet.getRange(c).setNumberFormat('0%');
  header(sheet,57,['模型','EP','P基线通信 ms','P争用后 ms','P forward增量','D基线通信 ms','D争用后 ms','D forward增量','P基线FW ms输入','P forward增幅','D基线FW ms输入','D forward增幅','P带宽 GB/s 输入','D带宽 GB/s 输入','评估范围']);
  for(let i=0;i<cases.length;i++) {
    const r=58+i,s=6+i;
    const valid=`AND(ISNUMBER(M${r}),M${r}>0,ISNUMBER(N${r}),N${r}>0,ISNUMBER($F$55),$F$55>=0,$F$55<1,ISNUMBER($H$55),$H$55>=0,$H$55<=1,ISNUMBER($J$55),$J$55>=0,$J$55<=1)`;
    fx(sheet,`A${r}`,`A${s}`);fx(sheet,`B${r}`,`B${s}`);
    fx(sheet,`M${r}`,'$B$55');fx(sheet,`N${r}`,'$D$55');
    fx(sheet,`C${r}`,`IF(${valid},I${s}*2^30/B${s}/(M${r}*1e6),NA())`);
    fx(sheet,`F${r}`,`IF(${valid},J${s}*2^30/B${s}/(N${r}*1e6),NA())`);
    fx(sheet,`D${r}`,`C${r}*(1+$H$55*$F$55/(1-$F$55))`);
    fx(sheet,`G${r}`,`F${r}*(1+$H$55*$F$55/(1-$F$55))`);
    fx(sheet,`E${r}`,`(D${r}-C${r})*$J$55`);fx(sheet,`H${r}`,`(G${r}-F${r})*$J$55`);
    fx(sheet,`J${r}`,`IF(AND(ISNUMBER(I${r}),I${r}>0,I${r}>=C${r}*$J$55),E${r}/I${r},"")`);
    fx(sheet,`L${r}`,`IF(AND(ISNUMBER(K${r}),K${r}>0,K${r}>=F${r}*$J$55),H${r}/K${r},"")`);
    put(sheet,`O${r}`,cases[i].profile==='deepseek_v41'?'仅EP；V4.1 PD条件评估':'仅EP；总耗时需实测基线');
  }
  const last=57+cases.length;
  sheet.getRange(`C58:I${last}`).setNumberFormat('0.000');sheet.getRange(`K58:K${last}`).setNumberFormat('0.000');
  for(const c of ['J','L'])sheet.getRange(`${c}58:${c}${last}`).setNumberFormat('0.00%');
  for(const c of ['I','K','M','N'])sheet.getRange(`${c}58:${c}${last}`).format.fill='#FFF2CC';
  sheet.getRange(`M58:N${last}`).setNumberFormat('0.00');
  put(sheet,`A${last+2}`,'带宽下降 10% 对受影响带宽项增加 11.11%。forward 增幅还需真实基线；不把 KV 交接等待再加到每个 Decode Step。');
  put(sheet,`A${last+4}`,'黄色 M/N 列可填各模型、各 EP 的无争用有效带宽，GB/s=10⁹ B/s。默认共享比例为 1；逐模型表可按事件调整。');
  return {first:58,last};
}
