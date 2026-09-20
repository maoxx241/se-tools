// Display-only changes requested for the EP/KV workbook. Numerical definitions
// remain documented in analysis/PD-MINUTE.md and are not changed by this module.
export function cleanSweepText(value) {
  if(typeof value!=='string')return value;
  if(value==='稳态相位平均估算。每层以等长 EP 通信段建模；未使用逐层 profiling，次数可为小数。')
    return '每层以等长 EP 通信段计算，次数按一分钟统计。';
  if(value==='估算')return '';
  return value.replaceAll('相位平均估算','').replaceAll('估算；','').replaceAll('"估算"','""')
    .replaceAll('本页估算','本页计算').replaceAll('；运行未验证','').replaceAll('运行未验证','')
    .replaceAll('；未实机验证','');
}

export function applyEditableSweepPresentation(wb,{sheetNames}={}) {
  const summary=[];
  for(const s of wb.worksheets.items) {
    if(sheetNames&&!sheetNames.includes(s.name))continue;
    const used=s.getUsedRange(),values=used.values,formulas=used.formulas;
    let changes=0;
    for(let i=0;i<values.length;i++)for(let j=0;j<values[i].length;j++) {
      const f=formulas[i]?.[j],v=values[i][j],next=cleanSweepText(f||v);
      if(next===(f||v))continue;
      if(f)used.getCell(i,j).formulas=[[next]];
      else used.getCell(i,j).values=[[next]];
      changes++;
    }
    s.showGridLines=true;
    used.format.font={name:'Carlito',size:11,color:'#000000'};
    used.format.rowHeight=18;
    // Keep input cues and the existing column geometry. Normal blank cells have
    // no fill/protection, so the native grid remains visible beyond all tables.
    const headerStarts=new Set(['模型','输入场景','输入有效','P GB/s']);
    for(let i=0;i<values.length;i++) {
      const row=values[i];
      if(headerStarts.has(row[0])) {
        const width=row.reduce((n,v,j)=>v!==null&&v!==''?j+1:n,0);
        s.getRangeByIndexes(i,0,1,width).format={fill:'#D9E1F2',font:{name:'Carlito',size:11,bold:true,color:'#000000'},
          borders:{preset:'all',style:'thin',color:'#A6A6A6'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:42};
      }
      if(row[2]==='每请求合计'||row[1]==='合计/请求')s.getRangeByIndexes(i,0,1,row.length).format.font.bold=true;
    }
    for(const c of s.name==='规格汇总'?['A2','A20','A51']:['A2'])s.getRange(c).format.font={name:'Carlito',size:14,bold:true,color:'#000000'};
    let inputs=[];
    if(s.name==='规格汇总')inputs=['B55','D55','F55','H55','J55','I58:I69','K58:K69','M58:N69'];
    if(s.name==='KV明细') {
      inputs=['B5','E5','H5','K5'];
      s.getRange(`A9:M${values.length}`).format.rowHeight=22;
    }
    if(s.name==='EP通信')inputs=['B5','D5','F5','H5','J5','L5'];
    if(s.name==='分钟场景') {
      inputs=['B5','F5','J5','N5','B7','F7','J7','N7','B9','F9','J9','N9','B11','B15:C17','F15:F18','I23:N34'];
      s.getRange('A:A').format.columnWidth=24;
      s.getRange('I:I').format.columnWidth=18;
      s.getRange('M:N').format.columnWidth=19;
      s.getRange('O:O').format.columnWidth=30;
      const cacheHeader=values.findIndex(row=>row[0]==='模型'&&row[2]==='组件')+1;
      if(cacheHeader)s.getRange(`A${cacheHeader+1}:M${values.length}`).format.rowHeight=45;
    }
    for(const c of inputs)s.getRange(c).format={fill:'#FFF2CC',borders:{preset:'all',style:'thin',color:'#D9D9D9'}};
    summary.push({sheet:s.name,textChanges:changes});
  }
  return summary;
}
