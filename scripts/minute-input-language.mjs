// Explain the existing assumptions without changing inputs or calculations.
export function clarifyMinuteInputs(s, region) {
  const edits=new Map();
  const put=(cell,text)=>{s.getRange(cell).values=[[text]];edits.set(cell,text);};
  for(const [cell,text] of [
    ['E5','P deployment\ncount'],['I5','D deployment\ncount'],
    ['A9','KV bursts\nper interval'],['E9','EP shared\nresource fraction'],['I9','EP exposed\ncomm fraction'],
    ['M22','EP shared\nresource fraction'],['N22','EP exposed\ncomm fraction']]) {
    put(cell,text);s.getRange(cell).format.wrapText=true;
  }
  const hasTp=s.getRange('P22').values[0][0]==='TP GB/s 输入';
  if(hasTp)for(const [cell,text] of [
    ['I11','TP shared\nresource fraction'],['M11','TP exposed\ncomm fraction'],
    ['Q22','TP shared\nresource fraction'],['R22','TP exposed\ncomm fraction']]) {
    put(cell,text);s.getRange(cell).format.wrapText=true;
  }
  // Existing labels and notes stay in the input panel; no hidden rows or new tab.
  put('D11','');
  put('A12','P = Prefill，D = Decode。通常只需修改带宽、请求数和 TTFT/TPOT；附加假设可保持默认。');
  put('A13','Total overhead 表示基线 60 秒工作量的额外耗时；TTFT 在这里作为 KV 批次到达间隔。');
  put('H15','Default: P = D = 1; KV bursts = 1; shared / exposed fractions = 100%.');
  put('H16','Deployment count：完整 EP 部署的套数；EP32 每套 32 卡，EP256 每套 256 卡。');
  put('H17','Shared resource fraction：通信中与 KV 竞争同一资源的比例。100% = 全部；0% = 资源独立。');
  put('H18','Exposed communication fraction：通信增时最终计入 forward 的比例；被计算遮住的部分不计入。');
  put('H19','KV bursts per interval：一批 KV 均分成 m 次、等间隔发送；m = 2 时每次一半，总量不变。');
  put('H20','Temporal overlap 由 KV 窗口自动计算；没有 profiling 时，不必另猜 shared / exposed 两个比例。');
  put(`L${region.first-1}`,'Total overhead\nms/60s');
  s.getRange(`L${region.first-1}`).format.wrapText=true;
  if(hasTp) {
    put(`S${region.first-1}`,'EP overhead\nms/60s');put(`T${region.first-1}`,'TP overhead\nms/60s');
    s.getRange(`S${region.first-1}:T${region.first-1}`).format.wrapText=true;
  }
  put(`V${region.buildFirst-1}`,'Total overhead\nms/60s');
  s.getRange(`V${region.buildFirst-1}`).format.wrapText=true;
  for(const row of [5,9,...(hasTp?[11]:[])])s.getRange(`A${row}:N${row}`).format.rowHeight=34;
  s.getRange('H15:U20').format.wrapText=false;
  return edits;
}
