// Formula events use local input payloads, except the explicitly labeled
// AlltoAllV rows, which are uniform cross-rank expectations.
export function v41WorkbookEvents(S, facts) {
  const T = S('stepTokens'), P = S('tp'), N = S('nodeRanks');
  const local = `ROUNDUP(${T}/${P},0)`, padded = `(${local}*${P})`;
  const A = S('activationBytes'), L = facts.layers, H = facts.H;
  const events = [];
  const add = (strategy, module, location, collective, elements, bytes, group, count) => events.push({ strategy, module, location, collective, elements: `=${elements}`, bytes: `=${bytes}`, group: `=${group}`, count: `=${count}` });
  add('SP', 'Attention', '逐层输入', 'Ring AllGather', `${local}*${H}`, A, P, `${L}*${S('sp')}`);
  add('DSA CP', 'Attention', 'Token/Head 交换', 'Equal-split AllToAll', `${local}*${facts.heads * facts.headDim}`, A, P, `${L}*${S('dsaCP')}`);
  add('SP', 'Attention', 'O Proj 后', 'Ring ReduceScatter', `${padded}*${H}`, A, P, `${L}*${S('sp')}`);
  add('Attention TP', 'Attention', '非 SP 输出', 'Ring AllReduce', `${T}*${H}`, A, P, `${L}*(1-${S('sp')})`);
  add('SP', 'Model Output', '最终 Hidden', 'Ring AllGather', `${local}*${H}`, A, P, S('sp'));
  // Current SP path uses replicated shared-expert linears, and Indexer runs
  // locally (including CP-local queries). Neither introduces its own gather.
  add('EP 期望', 'MoE Hidden', 'Dispatch 仅 Hidden', 'AllToAllV（跨 Rank 发送）', `${local}*${H}*${facts.topK}*${S('routeFraction')}`, S('dispatchBytes'), S('ep'), String(L));
  add('EP 期望', 'MoE Hidden', 'Combine 仅 Hidden', 'AllToAllV（跨 Rank 发送）', `${local}*${H}*${facts.topK}*${S('routeFraction')}`, S('combineBytes'), S('ep'), String(L));
  const on = S('engramEnabled'), tables = facts.engramTables;
  const leader = `(${T}*${facts.engramColumns}*${tables})`;
  add('Engram', 'Node Metadata', '两表合并 Counts', 'Ring AllGather', `${tables}*(${N}+1)`, '8', N, on);
  // Averaged across all node ranks; TP leaders submit, owners reply.
  add('Engram 期望', 'Node IDs', '均匀 Owner 路由', 'AllToAllV（跨 Rank 发送）', `${leader}/${P}*(${N}-1)/${N}`, '8', N, on);
  add('Engram 期望', 'Node Responses', '反向 BF16 返回', 'AllToAllV（跨 Rank 发送）', `${leader}/${P}*(${N}-1)/${N}*${facts.engramWidth}`, '2', N, on);
  add('Engram', 'TP Lookup', '两表合并广播', 'Tree Broadcast（Rank 均值）', `${leader}*${facts.engramWidth}`, '2', P, on);
  return events;
}

export function v41WorkbookNotes(sheet, S, config) {
  const c = config.text_config;
  sheet.getRange('O48:P48').values = [['VA 缓存 / Engram 存储', 'GiB / Rank']];
  const T = S('inputTokens'), R = S('cacheRequests');
  const longRows = `(3*INT(${T}/2)+${T})`;
  const formulas = [
    ['长 KV + Index（每 Rank）', `=${longRows}*(512*2+128+2)*${R}/2^30`],
    ['SWA 保留载荷（每 Rank）', `=40*MIN(${T},128)*512*2*${R}/2^30`],
    ['C2 Ring（每 Rank）', `=IF(${T}>0,3*32*1024*4*${R}/2^30,0)`],
    ['Engram BF16（节点内均摊）', `=${c.engram_num_embeddings.reduce((s, n) => s + n, 0)}*256*2/${S('nodeRanks')}/2^30`],
    ['Engram INT8+F32 scale 均摊', `=${c.engram_num_embeddings.reduce((s, n) => s + n, 0)}*(256+8*4)/${S('nodeRanks')}/2^30`],
    ['Engram FP8+E8M0（CPU 均摊）', `=${c.engram_num_embeddings.reduce((s, n) => s + n, 0)}*(256+8)/${S('nodeRanks')}/2^30`],
  ];
  for (let i = 0; i < formulas.length; i++) {
    sheet.getRange(`O${49 + i}`).values = [[formulas[i][0]]];
    sheet.getRange(`P${49 + i}`).formulas = [[formulas[i][1]]];
  }
  sheet.getRange('P49:P54').setNumberFormat('0.000000');
  const notes = [
    '权重使用官方源格式；不等于 VA 驻留字节。两阶段各含全部权重及 DSpark。',
    '通信只覆盖表中事件。未合计 LM Head、Embedding、draft、MoE 元数据或设备同步。',
    'Ring/Tree 是显式基准；EP/Owner 为均匀路由期望。不同通信域不合计。',
    'SP 下共享专家复制；Indexer 本地选 TopK。这两项无独立 Gather。',
    '缓存按输入长度×缓存案例请求数；表示完整请求保留载荷，非 Step 或 P→D 流量。',
    '长缓存全 TP 复制。物理分配含 padding；每全局 ID 为 540928 B，不能按层数再乘。',
    'Engram 三种存储为可选表示。默认返回 BF16；FP8/MXFP8 表在 CPU。',
    '范围：EP=TP×DP；TP 留在节点内；PP=PCP=DCP=1；SP=0 时本表仅建模 TP=1。',
    '投机步数默认 0。启用 DSpark 后，当前表仅扩大 target Token，draft 通信需另加。',
  ];
  for (let i = 0; i < notes.length; i++) {
    const row = 56 + i;
    sheet.getRange(`O${row}:P${row}`).merge();
    sheet.getRange(`O${row}`).values = [[notes[i]]];
    sheet.getRange(`O${row}:P${row}`).format = { wrapText: true, rowHeight: 46, font: { color: '#666666', size: 10 } };
  }
  sheet.getRange('O48:P48').format = { fill: '#D9E2F3', font: { bold: true } };
}
