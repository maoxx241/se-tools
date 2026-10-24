import { formats, resolveInputs, ruleInputs, stageOwnership } from '../lib/model-memory/engine.mjs';

export const inputRows = {
  tp: 2, kdaOTP: 3, attentionTP: 4, attentionOTP: 5, lmHeadTP: 6, embeddingTP: 7, otherTP: 8,
  ep: 9, sharedTP: 10, dsaCP: 11, dcp: 12, mlaDtype: 13, indexDtype: 14,
  convBytes: 16, recurrentBytes: 17, tokens: 18, blockSize: 19, specTokens: 20,
};
const blue = '#D9E1F2', yellow = '#FFF2CC', gray = '#D9D9D9';
const gib = 1073741824;
export function columnName(index) {
  let name = '';
  for (let n = index + 1; n; n = Math.floor((n - 1) / 26)) name = String.fromCharCode(65 + (n - 1) % 26) + name;
  return name;
}
const put = (s, address, formula) => { s.getRange(address).formulas = [[formula]]; };
const value = (s, address, data) => { s.getRange(address).values = [[data]]; };
const header = (s, range) => {
  s.getRange(range).format = { fill: blue, font: { name: 'Arial', size: 10, bold: true },
    borders: { bottom: { style: 'thin', color: '#A6A6A6' } } };
};
const block = (s, range) => { s.getRange(range).format.borders = { insideHorizontal: { style: 'thin', color: gray } }; };
const product = parts => parts.length ? parts.map(n => `(${n})`).join('*') : '1';

function byteFormulas(dims, dtypeCell) {
  const prefix = product(dims.slice(0, -1)), last = dims.at(-1), output = dims[0];
  return [
    `=${prefix}*ROUNDUP((${last})*VLOOKUP(${dtypeCell},$Q$32:$R$38,2,FALSE),0)`,
    `=IF(OR(${dtypeCell}="MXFP4",${dtypeCell}="MXFP8"),${prefix}*ROUNDUP((${last})/32,0),0)`,
    `=IF(${dtypeCell}="INT8",(${output})*8,IF(${dtypeCell}="INT4",(${output})*16,0))`,
  ];
}

function repeatForStage(row, owner, experts) {
  if (row.placement) return (row.placement === 'first' ? owner.first : owner.last) ? String(row.instances ?? 1) : '0';
  const layers = row.layerIds.filter(n => n >= owner.start && n <= owner.end).length;
  return row.rule === 'EP' ? `${layers}*ROUNDUP(${experts}/$R$9,0)` : String(layers);
}

function structureGroups(facts, scope) {
  const attention = r => ['DSA MLA', 'Gated MLA'].includes(r.group);
  const result = [
    ['embedding', o => Number(o.first), r => r.name === 'embed_tokens.weight'],
    ['lm_head', o => Number(o.last), r => r.name === 'lm_head.weight'],
    ['Decoder norms / residual', o => o.layers, r => r.group === 'Decoder common' && !r.placement],
    ['Final norm / residual', o => Number(o.last), r => r.group === 'Decoder common' && Boolean(r.placement)],
    ['Dense MLP', o => o.dense, r => r.group === 'Dense MLP'],
    ['KDA', o => o.kda, r => r.group === 'KDA'],
    [facts.hasDSA ? 'DSA MLA' : 'Gated MLA', o => o.attention, attention],
  ];
  if (facts.hasDSA) result.push(['DSA Indexer', o => o.attention, r => r.group === 'DSA Indexer']);
  result.push(
    ['MoE router / latent', o => o.moe, r => r.group === 'MoE' && r.rule !== 'EP' && !r.shared],
    ['Routed experts', o => o.moe, r => r.rule === 'EP'],
    ['Shared experts', o => o.moe, r => r.shared],
  );
  if (scope === 'full') result.push(
    ['ViT blocks', o => o.first ? facts.visionLayers : 0, r => r.group === 'ViT blocks'],
    ['ViT patch / final / MM', o => Number(o.first), r => r.group.startsWith('ViT') && r.group !== 'ViT blocks' || r.group === 'MM projector'],
  );
  return result;
}

export function buildMemorySheet(workbook, model, item) {
  const { preset, rows } = item, { facts } = model.spec;
  const p = resolveInputs(preset), ownership = stageOwnership(facts, preset.partitions), pp = ownership.length;
  const s = workbook.worksheets.add(preset.name), end = rows.length + 1;
  const stageHeaderRow = Math.max(86, end + 3), stageFirstRow = stageHeaderRow + 1;
  const maxRow = Math.max(92, stageFirstRow + pp + 1), finalRow = Math.max(98, maxRow + 3);
  const stageCols = ownership.map((_, i) => columnName(26 + i));
  const helperCols = Array.from({ length: 6 }, (_, i) => columnName(26 + pp + i));
  s.showGridLines = true;
  s.getRange(`A1:${helperCols.at(-1)}${finalRow}`).format = {
    font: { name: 'Arial', size: 10 }, verticalAlignment: 'center', rowHeight: 18,
  };
  s.getRange(`A1:O${end}`).values = [
    ['结构', '权重', '权重维度', '数据类型', '全模型重复数', '单实例参数量', 'TP 策略', 'EP 策略', '共享专家',
      '全模型权重 GiB', '本 stage 单卡 GiB', '量化规则', '切分份数', '单卡权重维度', '单卡重复数'],
    ...rows.map(r => [r.group, r.name, JSON.stringify(r.shape), r.dtype, r.repeat, null, r.rule,
      r.rule === 'EP' ? 'EP' : 'None', r.shared ? '是' : '否', null, null, null, null, null, null]),
  ];
  s.getRange('Q1:T22').values = [
    ['并行配置', '值', '模型参数', '参数值'],
    ['KDA 主体 TP', p.tp, '本 stage KDA 层数', null],
    ['KDA o_proj TP', p.kdaOTP, '本 stage Attention 层数', null],
    ['Attention 主体 TP', p.attentionTP, 'KDA heads', facts.kdaHeads],
    ['Attention o_proj TP', p.attentionOTP, 'KDA head dim', facts.headDim],
    ['lm_head TP', p.lmHeadTP, 'Attention heads', facts.attentionHeads],
    ['embedding TP', p.embeddingTP, 'Indexer query heads', facts.indexHeads],
    ['其他语言 TP', p.otherTP, '本 stage MoE 层数', null],
    ['EP', p.ep, 'Routed experts / layer', facts.experts],
    ['共享专家 TP', p.sharedTP, 'Shared experts / layer', facts.sharedExperts],
    ['DSA CP', p.dsaCP, 'MLA latent dim', facts.kvRank],
    ['DCP', p.dcp, 'RoPE dim', facts.ropeDim],
    ['MLA cache dtype', p.mlaDtype, 'Indexer key dim', facts.indexDim],
    ['Indexer cache dtype', p.indexDtype, 'Conv kernel', facts.convKernel],
    ['Indexer scale bytes', null, 'Cache block size', null],
    ['KDA conv bytes', p.convBytes, '单请求 token 数', null],
    ['KDA recurrent bytes', p.recurrentBytes, '阶段', preset.phase],
    ['单请求 token 数', p.tokens, 'ViT', preset.scope === 'full' ? '包含' : '不含'],
    ['Cache block size', p.blockSize, 'KDA local heads', null],
    ['Speculative tokens', p.specTokens, 'Attention projection heads', null],
    ['主缓存 context 切分', null, 'Indexer key heads', facts.hasDSA ? 1 : 0],
    ['当前 PP stage', 0, 'Attention compute heads', null],
  ];
  header(s, 'A1:O1'); header(s, 'Q1:T1'); block(s, 'Q2:T22');
  s.getRange('R2:R22').format.fill = yellow;
  for (const [key, source] of Object.entries(preset.links ?? {})) {
    put(s, `R${inputRows[key]}`, `=$R$${inputRows[source.key]}*${source.multiplier ?? 1}`);
  }
  if (preset.attentionTPFromCP) put(s, 'R4', '=IF(R11>1,1,R2)');
  put(s, 'R15', facts.hasDSA ? '=IF(R14="BF16",0,IF(R14="INT8",2,4))' : '=0');
  put(s, 'R21', '=R12');
  put(s, 'T15', '=R19'); put(s, 'T16', '=R18'); put(s, 'T19', '=T4/R2');
  put(s, 'T20', '=T6/R4'); put(s, 'T22', '=T20*R12');
  for (const [key, field] of [[2, 'kda'], [3, 'attention'], [8, 'moe']]) {
    put(s, `T${key}`, `=CHOOSE(R22+1,${ownership.map(o => o[field]).join(',')})`);
  }
  s.getRange('R2:R12').dataValidation = { rule: { type: 'whole', operator: 'between', formula1: 1, formula2: 65536 } };
  s.getRange('R18').dataValidation = { rule: { type: 'whole', operator: 'between', formula1: 1, formula2: facts.maxTokens } };
  s.getRange('R19').dataValidation = { rule: { type: 'whole', operator: 'between', formula1: 1, formula2: facts.maxTokens } };
  s.getRange('R20').dataValidation = { rule: { type: 'whole', operator: 'between', formula1: 0, formula2: 64 } };
  s.getRange('R22').dataValidation = { rule: { type: 'list', values: ownership.map(o => String(o.stage)) } };
  s.getRange('R13').dataValidation = { rule: { type: 'list', values: facts.hasDSA ? ['BF16', 'INT8', 'FP8'] : ['BF16'] } };
  s.getRange('R14').dataValidation = { rule: { type: 'list', values: facts.hasDSA ? ['BF16', 'INT8', 'FP8'] : ['BF16'] } };
  s.getRange('R15').format.fill = '#F2F2F2'; s.getRange('R21').format.fill = '#F2F2F2';
  s.getRange(preset.phase === 'Prefill' ? 'R12' : 'R11').format.fill = '#F2F2F2';
  if (!facts.hasDSA) s.getRange('R11:R14').format.fill = '#F2F2F2';
  s.getRange('R2:R22').format.numberFormat = '#,##0';
  s.getRange('R13:R14').format.numberFormat = 'General';
  s.mergeCells('U1:V2'); value(s, 'U1', 'Made by mindie-agent');
  s.getRange('U1:V2').format = { fill: blue, font: { name: 'Arial', size: 11, bold: true, color: '#44546A' },
    horizontalAlignment: 'center', verticalAlignment: 'center' };
  s.getRange('S23:T28').values = [
    ['组网', '值'], ['DP', null], ['PP', pp], ['EP / stage', null], ['总设备数', null], ['DCP', null],
  ];
  put(s, 'T24', '=R9/R2'); put(s, 'T26', '=R9'); put(s, 'T27', '=R9*T25'); put(s, 'T28', '=R12');
  header(s, 'S23:T23'); block(s, 'S24:T28');
  s.getRange('Q31:R38').values = [['主权重类型', '主权重 bytes/元素'], ...Object.entries(formats).map(([name, f]) => [name, f.bytes])];
  header(s, 'Q31:R31'); block(s, 'Q32:R38'); s.getRange('R32:R38').format.numberFormat = '0.00000';
  s.getRange('S31:V35').values = [
    ['Cache 单 token', '元素 bytes', 'Scale bytes', '合计 bytes'],
    ['MLA latent', null, null, null], ['RoPE', 2, 0, null], ['MLA 合计', null, null, null],
    ['Indexer', null, null, null],
  ];
  put(s, 'T32', '=IF(R13="BF16",2,1)'); put(s, 'U32', '=IF(R13="BF16",0,ROUNDUP(T11/128,0)*4)');
  put(s, 'V32', '=T11*T32+U32'); put(s, 'V33', '=T12*T33'); put(s, 'V34', '=SUM(V32:V33)');
  put(s, 'T35', '=IF(R14="BF16",2,1)'); put(s, 'U35', '=R15'); put(s, 'V35', '=T13*T35+U35');
  header(s, 'S31:V31'); block(s, 'S32:V35');
  const valid = ['R2>=1', 'R9>=1', 'R12>=1', 'R19>=1', 'R18>=1', `R18<=${facts.maxTokens}`,
    'MOD(R9,R2)=0', 'MOD(T4,R2)=0', 'MOD(R2,R12)=0', `R22>=0`, `R22<${pp}`, 'MOD(R22,1)=0',
    'R20>=0', 'MOD(R20,1)=0', preset.phase === 'Prefill' ? 'R12=1' : 'R11=1',
    facts.hasDSA ? 'OR(R11=1,R11=R2)' : 'AND(R11=1,R13="BF16",R14="BF16")',
    ...Object.entries(inputRows).filter(([key]) => !['mlaDtype', 'indexDtype', 'specTokens'].includes(key))
      .flatMap(([, n]) => [`R${n}>=1`, `MOD(R${n},1)=0`]),
    ...new Set(rows.flatMap(row => ruleInputs[row.rule]
      ? [`MOD(${row.shape[row.axis]},R${inputRows[ruleInputs[row.rule]]})=0`] : [])),
  ];
  const validChunks = [];
  for (let i = 0; i < valid.length; i += 200) validChunks.push(`AND(${valid.slice(i, i + 200).join(',')})`);
  value(s, 'Q30', '参数校验（1=有效）'); put(s, 'R30', `=IF(AND(${validChunks.join(',')}),1,0)`);
  s.getRange('R30').conditionalFormats.add('cellIs', {
    operator: 'equal', formula: 0, format: { fill: '#FCE4D6', font: { color: '#9C0006', bold: true } },
  });
  value(s, 'S30', '量化方案'); value(s, 'T30', preset.recipe);

  for (const [i, row] of rows.entries()) {
    const n = i + 2, key = ruleInputs[row.rule], div = key ? `$R$${inputRows[key]}` : '1';
    const localDims = row.shape.map((d, axis) => axis === row.axis ? `${d}/M${n}` : String(d));
    put(s, `F${n}`, `=${product(row.shape)}`); put(s, `M${n}`, `=${div}`);
    const shape = localDims.map(d => `"&(${d})&"`).join(',');
    put(s, `N${n}`, `="[${shape}]"`);
    put(s, `O${n}`, `=CHOOSE($R$22+1,${ownership.map(o => repeatForStage(row, o, facts.experts)).join(',')})`);
    for (const [j, f] of byteFormulas(localDims, `D${n}`).entries()) put(s, `${helperCols[j]}${n}`, f);
    for (const [j, f] of byteFormulas(row.shape.map(String), `D${n}`).entries()) put(s, `${helperCols[j + 3]}${n}`, f);
    put(s, `J${n}`, `=E${n}*SUM(${helperCols[3]}${n}:${helperCols[5]}${n})/${gib}`);
    for (const [stage, col] of stageCols.entries()) {
      put(s, `${col}${n}`, `=IF($R$30=1,SUM(${helperCols[0]}${n}:${helperCols[2]}${n})*(${repeatForStage(row, ownership[stage], facts.experts)})/${gib},NA())`);
    }
    put(s, `K${n}`, pp === 1 ? `=${stageCols[0]}${n}` : `=INDEX(${stageCols[0]}${n}:${stageCols.at(-1)}${n},1,$R$22+1)`);
    put(s, `L${n}`, `=IF(OR(D${n}="MXFP4",D${n}="MXFP8"),"group 32 / E8M0",IF(D${n}="INT8","channel / 8 B",IF(D${n}="INT4","channel / 16 B","")))`);
  }
  s.getRange(`D2:D${end}`).dataValidation = { rule: { type: 'list', values: Object.keys(formats) } };
  s.getRange(`D2:D${end}`).format.fill = yellow;
  s.getRange(`${stageCols[0]}1:${helperCols.at(-1)}1`).values = [[
    ...ownership.map(o => `PP${o.stage} 单卡 GiB`), '本卡主权重 bytes', '本卡 scale bytes', '本卡附加 bytes',
    '全局主权重 bytes', '全局 scale bytes', '全局附加 bytes',
  ]];
  header(s, `${stageCols[0]}1:${helperCols.at(-1)}1`);
  const groups = structureGroups(facts, preset.scope), groupEnd = 39 + groups.length, groupTotal = groupEnd + 1;
  s.getRange(`Q39:X${groupEnd}`).values = [
    ['结构', '单卡层数', '每层实例数', '单卡重复总数', '每实例 GiB', '单卡合计 GiB', '占权重比例', '组成'],
    ...groups.map(([label]) => [label, null, null, null, null, null, null, label === 'Shared experts' ? `${facts.sharedExperts} experts 融合单元` : '']),
  ];
  for (const [i, [label, count, predicate]] of groups.entries()) {
    const n = i + 40;
    put(s, `R${n}`, `=CHOOSE($R$22+1,${ownership.map(count).join(',')})`);
    put(s, `S${n}`, label === 'Routed experts' ? '=ROUNDUP($T$9/$R$9,0)' : '=1');
    put(s, `T${n}`, `=R${n}*S${n}`);
    const matched = rows.flatMap((r, i) => predicate(r) ? [`K${i + 2}`] : []);
    put(s, `V${n}`, matched.length ? `=SUM(${matched.join(',')})` : '=0');
    put(s, `U${n}`, `=IF(T${n}=0,0,V${n}/T${n})`); put(s, `W${n}`, `=V${n}/$R$23`);
  }
  value(s, `Q${groupTotal}`, '合计'); put(s, `V${groupTotal}`, `=SUM(V40:V${groupEnd})`);
  header(s, 'Q39:X39'); block(s, `Q40:X${groupTotal}`); header(s, `Q${groupTotal}:X${groupTotal}`);
  s.getRange(`U40:V${groupTotal}`).format.numberFormat = '0.0000'; s.getRange(`W40:W${groupEnd}`).format.numberFormat = '0.0%';
  s.getRange('Q56:R58').values = [['运行 buffer', 'GiB'], ['Attention o_proj full gather', null], ['Attention o_proj transpose shard', null]];
  const oi = rows.findIndex(r => ['DSA MLA', 'Gated MLA'].includes(r.group) && r.name === 'o_proj') + 2;
  put(s, 'R57', facts.hasDSA && preset.phase === 'Prefill'
    ? `=IF(AND(R11>1,T3>0),SUM(${helperCols[3]}${oi}:${helperCols[4]}${oi})/${gib},0)` : '=0');
  put(s, 'R58', facts.hasDSA && preset.phase === 'Prefill' ? `=IF(AND(R11>1,OR(D${oi}="BF16",D${oi}="FP32")),K${oi},0)` : '=0');
  header(s, 'Q56:R56'); s.getRange('R57:R58').format.numberFormat = '0.0000';
  s.getRange('Q23:R29').values = [
    ['单卡权重 GiB', null], ['权重 + cache GiB', null], ['o_proj buffer GiB', null],
    ['总参数量 T', null], ['单请求 cache GiB', null], ['权重 + cache + buffer GiB', null], ['本页 token 数', null],
  ];
  put(s, 'R23', `=SUM(K2:K${end})`); put(s, 'R24', '=R23+R27'); put(s, 'R25', '=SUM(R57:R58)');
  put(s, 'R26', `=SUMPRODUCT(E2:E${end},F2:F${end})/1000000000000`); put(s, 'R27', '=X63');
  put(s, 'R28', '=R23+R27+R25'); put(s, 'R29', '=R18'); block(s, 'Q23:R29');
  s.getRange('R23:R28').format.numberFormat = '0.0000';
  s.getRange('Q62:Y72').values = [
    ['Token 数', '本卡主缓存 token', 'MLA latent+scale GiB', 'RoPE GiB', 'Indexer K+scale GiB',
      'KDA recurrent GiB', 'KDA conv GiB', '单请求 cache GiB', '权重 + cache GiB'],
    [null, null, null, null, null, null, null, null, null],
    ...(facts.maxTokens >= 1048576 ? [1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576]
      : Array.from({ length: 9 }, (_, i) => Math.max(1, Math.floor(facts.maxTokens / 2 ** (8 - i)))))
      .map(t => [t, null, null, null, null, null, null, null, null]),
  ];
  put(s, 'Q63', '=$R$18');
  for (let n = 63; n <= 72; n++) {
    put(s, `R${n}`, `=ROUNDUP(Q${n}/($R$19*$R$12),0)*$R$19`);
    put(s, `S${n}`, `=R${n}*$T$3*$V$32/${gib}`); put(s, `T${n}`, `=R${n}*$T$3*$V$33/${gib}`);
    put(s, `U${n}`, `=R${n}*$R$12*$T$3*$V$35/${gib}`);
    put(s, `V${n}`, `=$T$2*($T$4/$R$2)*$T$5*$T$5*$R$17/${gib}`);
    put(s, `W${n}`, `=$T$2*3*($T$4/$R$2)*$T$5*($T$14-1+$R$20)*$R$16/${gib}`);
    put(s, `X${n}`, `=IF($R$30=1,SUM(S${n}:W${n}),NA())`); put(s, `Y${n}`, `=$R$23+X${n}`);
  }
  header(s, 'Q62:Y62'); block(s, 'Q63:Y72'); s.getRange('Q64:Q72').format.fill = yellow;
  s.getRange('Q63:R72').format.numberFormat = '#,##0'; s.getRange('S63:Y72').format.numberFormat = '0.0000';
  s.getRange('Q74:R82').values = [
    ['缓存组成', '本卡布局'], ['MLA 主缓存', null], ['RoPE', 'BF16'],
    ['Indexer K', facts.hasDSA ? 'DCP 组内按卡复制' : '不适用'],
    ['KDA recurrent', 'FP32 / head TP'], ['KDA conv', 'BF16 / head TP'],
    ['阶段', facts.hasDSA ? preset.phase === 'Prefill' ? 'DSA CP 切计算，主 KV 按卡保存' : 'MLA 按 DCP 切 token，Indexer 按卡复制' : 'Gated MLA BF16，KDA 按 head TP'],
    ['Indexer scale', null], ['Cache dtype', null],
  ];
  put(s, 'R75', '="[local_blocks, "&R19&", 1, "&V34&" bytes]"');
  put(s, 'R81', facts.hasDSA ? '=IF(R14="BF16","无",IF(R14="INT8","FP16 / token","FP32 / token"))' : '="不适用"');
  put(s, 'R82', '="MLA "&R13&" / Indexer "&R14'); header(s, 'Q74:R74');
  s.getRange(`Q${stageHeaderRow}:AC${stageHeaderRow + pp}`).values = [
    ['PP stage', '层范围', '层数', 'KDA 层', 'Attention 层', 'MoE 层', '权重 GiB', 'Cache GiB', '合计 GiB',
      'embedding', 'lm_head', 'ViT / MM', 'Buffer GiB'],
    ...ownership.map(o => [o.stage, `${o.start}-${o.end}`, o.layers, o.kda, o.attention, o.moe,
      null, null, null, Number(o.first), Number(o.last), preset.scope === 'full' ? Number(o.first) : 0, null]),
  ];
  for (const [i, col] of stageCols.entries()) {
    const n = stageFirstRow + i;
    put(s, `W${n}`, `=SUM(${col}2:${col}${end})`);
    put(s, `X${n}`, `=IF($R$30=1,ROUNDUP($R$18/($R$19*$R$12),0)*$R$19*U${n}*($V$34+$R$12*$V$35)/${gib}+T${n}*($T$4/$R$2)*$T$5*$T$5*$R$17/${gib}+T${n}*3*($T$4/$R$2)*$T$5*($T$14-1+$R$20)*$R$16/${gib},NA())`);
    put(s, `AC${n}`, facts.hasDSA && preset.phase === 'Prefill'
      ? `=IF($R$11>1,IF(U${n}>0,SUM(${helperCols[3]}${oi}:${helperCols[4]}${oi})/${gib},0)+IF(OR($D$${oi}="BF16",$D$${oi}="FP32"),${col}${oi},0),0)` : '=0');
    put(s, `Y${n}`, `=W${n}+X${n}+AC${n}`);
  }
  value(s, `Q${maxRow}`, '最大单卡');
  for (const col of ['W', 'X', 'Y', 'AC']) put(s, `${col}${maxRow}`, `=MAX(${col}${stageFirstRow}:${col}${stageHeaderRow + pp})`);
  header(s, `Q${stageHeaderRow}:AC${stageHeaderRow}`); block(s, `Q${stageFirstRow}:AC${stageHeaderRow + pp}`); header(s, `Q${maxRow}:AC${maxRow}`);
  s.getRange(`W${stageFirstRow}:Y${maxRow}`).format.numberFormat = '0.0000'; s.getRange(`AC${stageFirstRow}:AC${maxRow}`).format.numberFormat = '0.0000';
  const widths = [23, 36, 26, 12, 14, 20, 18, 12, 12, 20, 22, 23, 14, 30, 16, 3, 32, 22, 28, 29, 26, 24, 24, 28, 26, 18];
  widths.forEach((width, i) => { s.getRangeByIndexes(0, i, finalRow, 1).format.columnWidth = width; });
  s.getRange(`${stageCols[0]}1:${helperCols.at(-1)}${finalRow}`).format.columnWidth = 24;
  s.getRange(`${stageCols[0]}2:${stageCols.at(-1)}${end}`).format.numberFormat = '0.000000';
  s.getRange(`${helperCols[0]}2:${helperCols.at(-1)}${end}`).format.numberFormat = '#,##0';
  s.getRange(`E2:F${end}`).format.numberFormat = '#,##0'; s.getRange(`J2:K${end}`).format.numberFormat = '0.000000';
  s.getRange(`M2:M${end}`).format.numberFormat = '0'; s.getRange(`O2:O${end}`).format.numberFormat = '#,##0';
  s.getRange('A1:O1').format.wrapText = true; s.getRange('Q62:Y62').format.wrapText = true;
  s.getRange('A1:O1').format.rowHeight = 32; s.getRange('Q62:Y62').format.rowHeight = 32;
  block(s, `A2:O${end}`);
  let previous = '';
  rows.forEach((r, i) => {
    if (previous !== r.group) s.getRange(`A${i + 2}:O${i + 2}`).format.borders = { top: { style: 'medium', color: '#999999' } };
    previous = r.group;
  });
  s.freezePanes.freezeRows(1); s.freezePanes.freezeColumns(2);
  return { sheet: s, preset, rows, ownership, end, groupTotal, stageCols, helperCols, stageHeaderRow, stageFirstRow, maxRow };
}

export function createMemoryWorkbook(model, Workbook) {
  const workbook = Workbook.create();
  const sheets = model.scenarios.map(item => buildMemorySheet(workbook, model, item));
  workbook.recalculate();
  return { workbook, sheets };
}
