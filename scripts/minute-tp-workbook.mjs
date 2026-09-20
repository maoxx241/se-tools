// Targeted extension of the existing minute sheet. Keep user inputs and cache
// formulas in place; EP and TP bandwidth terms have separate controls.
export function addMinuteTp(s, region, items) {
  const put=(c,v)=>{s.getRange(c).values=[[v]];};
  const fx=(c,f)=>{s.getRange(c).formulas=[[`=${f}`]];};
  const head=(cell,labels,source)=>{
    const r=s.getRange(cell).resize(1,labels.length);
    r.copyFrom(s.getRange(source),'all');r.values=[labels];
    r.format={fill:'#D9E1F2',font:{name:'Carlito',size:11,bold:true,color:'#000000'},
      borders:{preset:'all',style:'thin',color:'#A6A6A6'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:42};
  };
  const {first,last,buildFirst,buildLast,parameterFirst,parameterLast}=region;
  const oh=first-1,bh=buildFirst-1;
  const alreadyTp=s.getRange(`P${parameterFirst-1}`).values[0][0]==='TP GB/s 输入';
  // Re-running updates formulas, but never resets a user's new TP inputs.
  for(const [label,c,text,v,format] of [
    ['E11','F11','TP 带宽 GB/s',4800,'#,##0.00'],
    ['I11','J11','TP 共享比例',1,'0.0%'],['M11','N11','TP 暴露率',1,'0.0%']]) {
    put(label,text);
    if(!alreadyTp) {
      s.getRange(c).copyFrom(s.getRange('N7'),'all');put(c,v);s.getRange(c).setNumberFormat(format);
    }
    s.getRange(c).format={fill:'#FFF2CC',borders:{preset:'all',style:'thin',color:'#D9D9D9'}};
  }
  put('D11','');
  put('A12','TTFT 作为批次周期；EP 与 TP 带宽分别填写。TP 不与 KV 共用链路时，将 TP 共享比例设为 0。');
  put('E9','EP 共享比例');put('I9','EP 暴露率');
  put('A3','按层分布等效 EP、TP 通信段，次数按一分钟统计。');
  put('H18','全通信＝该步所有等效 EP＋TP 段都在 KV 窗口内；TP 首尾通信均摊到层段。');
  head(`P${parameterFirst-1}`,['TP GB/s 输入','TP 共享比例','TP 暴露率','Attention 层数','Dense 层数','共享专家 TP 层数'],`I${parameterFirst-1}`);
  for(let i=0;i<items.length;i++) {
    const p=parameterFirst+i,m=items[i];
    for(const [c,ref] of [['P','$F$11'],['Q','$J$11'],['R','$N$11']]) {
      const value=s.getRange(`${c}${p}`).values[0][0];
      if(!alreadyTp&&!s.getRange(`${c}${p}`).formulas[0][0]&&(value===null||value==='')) {
        s.getRange(`${c}${p}`).copyFrom(s.getRange(`I${p}`),'all');fx(`${c}${p}`,ref);
      }
    }
    s.getRange(`P${p}:R${p}`).format={fill:'#FFF2CC',borders:{preset:'all',style:'thin',color:'#D9D9D9'}};
    s.getRange(`P${p}`).setNumberFormat('#,##0.00');
    s.getRange(`Q${p}:R${p}`).setNumberFormat('0.0%');
    s.getRange(`S${p}:U${p}`).values=[[m.facts.attentionLayers,m.facts.denseLayers||0,m.profile==='kimi'?m.facts.moeLayers:0]];
  }
  head(`Q${oh}`,['TP GB/Rank/步','TP 带宽项 ms','EP 增时 ms/60s','TP 增时 ms/60s'],`P${oh}`);
  put(`H${oh}`,'EP+TP 重叠 FW/min');put(`I${oh}`,'全通信 FW/min');put(`J${oh}`,'部分通信 FW/min');
  put(`K${oh}`,'重叠层段 次/min');put(`P${oh}`,'最低 EP GB/s（扣 TP）');
  put(`A${bh-2}`,'EP、TP 增时分别计算后相加；重叠 FW 统计合并层段，不能把两种通信的 FW 次数相加。');
  put(`S${bh}`,'全通信概率');
  head(`Y${bh}`,['Embedding TP GB','Attention SP GB','Dense TP GB','Shared TP GB','Final Hidden GB','Sampling TP GB',
    'TP GB/Rank/步','TP 带宽项 ms','每层 TP 段 ms','TP 每层增时 ms','EP 增时 ms/60s','TP 增时 ms/60s','EP+TP 带宽 ms','每层合并段 ms'],`N${bh}`);
  for(let i=0;i<items.length;i++)for(let j=0;j<12;j++) {
    const p=parameterFirst+i,r=first+i*12+j,b=buildFirst+i*12+j;
    const TP=`D${p}`,T=`K${p}`,H=`F${p}`,L=`E${p}`,n=`ROUNDUP(${T}/${TP},0)`,d='$J$7',a=`(1-${d})`;
    const old=s.getRange(`A${b}`).formulas[0][0].replace(/^=/,'');
    if(!alreadyTp)fx(`A${b}`,`AND(${old},ISNUMBER(P${p}),P${p}>0,ISNUMBER(Q${p}),Q${p}>=0,Q${p}<=1,ISNUMBER(R${p}),R${p}>=0,R${p}<=1)`);
    const calc=(c,f)=>fx(`${c}${b}`,`IF(A${b},${f},"")`);
    calc('Y',`${T}*${H}*2*2*(${TP}-1)/${TP}/1e9`);
    calc('Z',`S${p}*2*${n}*${H}*2*(${TP}-1)/1e9`);
    calc('AA',`T${p}*${n}*${H}*2*2*(${TP}-1)/${TP}/1e9`);
    calc('AB',`U${p}*2*${n}*${H}*2*(${TP}-1)/1e9`);
    calc('AC',`${n}*${H}*2*(${TP}-1)/1e9`);
    calc('AD',`${T}*256*(2+4)*(${TP}-1)/1e9`);
    calc('AE',`SUM(Y${b}:AD${b})`);calc('AF',`AE${b}*1000/P${p}`);calc('AG',`AF${b}/${L}`);
    calc('AK',`O${b}+AF${b}`);calc('AL',`AK${b}/${L}`);
    calc('Q',`IF(C${b}>AF${b},N${b}*1000/(C${b}-AF${b}),"")`);
    fx(`W${b}`,`IF(NOT(A${b}),"输入无效",IF(K${b}>=L${b},"KV 超载",IF(AK${b}>C${b}+1e-9,"TPOT 不自洽",IF(OR(C${b}>L${b},MAX(P${b},AG${b})>L${b}-K${b}),"需时序仿真",""))))`);
    const ok=`W${b}=""`,gap=`(C${b}-AK${b})/${L}`,span=`((${L}-1)*C${b}/${L}+AL${b})`;
    const probability=w=>`1-((${L}-1)*MAX(0,${gap}-(${w}))+MAX(0,L${b}-${span}-(${w})))/L${b}`;
    fx(`R${b}`,`IF(${ok},IF(OR(K${b}=0,AK${b}=0),0,${probability(`K${b}`)}),"")`);
    fx(`S${b}`,`IF(${ok},IF(AK${b}=0,0,MAX(0,1-(${probability(`L${b}-K${b}`)}))),"")`);
    for(const [c,event] of [['T',`P${b}`],['AH',`AG${b}`]])
      fx(`${c}${b}`,`IF(${ok},IF(OR(${event}=0,K${b}=0,${d}=0),0,IF(${event}<=${a}*K${b},${d}*${event}*K${b}/${a}-${d}^2*${event}^2/(2*${a}^2),${d}*${event}*K${b}+${d}^2*K${b}^2/2)/L${b}),"")`);
    fx(`AI${b}`,`IF(${ok},U${b}*${L}*T${b}*M${p}*N${p},"")`);
    fx(`AJ${b}`,`IF(${ok},U${b}*${L}*AH${b}*Q${p}*R${p},"")`);
    fx(`V${b}`,`IF(${ok},SUM(AI${b}:AJ${b}),"")`);fx(`X${b}`,`IF(${ok},U${b}*M${b},"")`);
    for(const [c,f] of [['H',`U${b}*R${b}`],['I',`U${b}*S${b}`],['J',`U${b}*MAX(0,R${b}-S${b})`],
      ['K',`IF(OR(AK${b}=0,K${b}=0),0,U${b}*${L}*MIN(1,(K${b}+AL${b})/L${b}))`],
      ['L',`V${b}`],['M',`E${r}*V${b}/60`],['N',`V${b}/60000*1e6`],
      ['S',`AI${b}`],['T',`AJ${b}`]])fx(`${c}${r}`,`IF(${ok},${f},"")`);
    fx(`O${r}`,`IF(${ok},"${items[i].cache.pullBytes===null?'V4.1 规划':''}",W${b})`);
    fx(`Q${r}`,`AE${b}`);fx(`R${r}`,`AF${b}`);
  }
  s.getRange(`P${parameterFirst}:U${parameterLast}`).format.font={name:'Carlito',size:11,color:'#000000'};
  s.getRange(`Q${first}:T${last}`).format.font={name:'Carlito',size:11,color:'#000000'};
  s.getRange(`Y${buildFirst}:AL${buildLast}`).format.font={name:'Carlito',size:11,color:'#000000'};
  s.getRange(`Q${first}:T${last}`).setNumberFormat('0.000000');
  s.getRange(`Y${buildFirst}:AL${buildLast}`).setNumberFormat('0.000000');
  s.getRange('P:AL').format.columnWidth=19;
  // Update the old scope note in place without changing unrelated cache rows.
  const range=s.getUsedRange(),values=range.values;
  for(let i=0;i<values.length;i++)if(typeof values[i][0]==='string'&&values[i][0].startsWith('MC2：'))
    put(`A${i+1}`,'MC2：A3 FullMesh BF16；TP/SP：HCCL Ring，K3 共享专家 TP、Qwen 共享专家 DP。通信启动、Engram 与排队另计。');
  return region;
}
