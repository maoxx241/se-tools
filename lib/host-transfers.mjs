// Host/device payload accounting. No collective/link-protocol multiplier.
import { readModelConfig } from './model-catalog.mjs';
import { communicationModels } from './communication-requirements.mjs';
import { v41Topology } from './deepseek-v41.mjs';

export const HOST_TRANSFER_DEFAULTS = Object.freeze({
  lengths: [16384, 262144, 1048576], eps: [32, 256], requestsPerDP: 16,
  prefillTP: 8, decodeTP: 1, ranksPerNode: 8, prefillChunkTokens: 16384,
  decodeTokensPerDP: 128, cpuVectorFraction: 1, vectorElementBytes: 2,
  metadataOnDevice: 1, blockSize: 128, speculativeConvSlots: 0,
  prefixHitFraction: 1, localRequestFraction: 0,
});

function positiveInt(x, name) {
  if (!Number.isSafeInteger(x) || x <= 0) throw new Error(`${name} must be a positive integer`);
}
function fraction(x, name) {
  if (!Number.isFinite(x) || x < 0 || x > 1) throw new Error(`${name} must be in [0,1]`);
}
export function engramGeometry(config) {
  const c = config.text_config || config;
  return { layers: c.engram_layer_ids.length, ngram: c.engram_max_ngram_size,
    heads: c.engram_n_heads, width: c.engram_head_dim };
}

export function engramTransfer(geometry, { ep, tp, tokensPerDP, rounds = 1, ...options }) {
  const o = { ...HOST_TRANSFER_DEFAULTS, ...options };
  for (const [k,v] of Object.entries({ep,tp,rounds,ranksPerNode:o.ranksPerNode})) positiveInt(v,k);
  if (!Number.isSafeInteger(tokensPerDP) || tokensPerDP < 0) throw new Error('Invalid tokensPerDP');
  if (ep % o.ranksPerNode || o.ranksPerNode % tp) throw new Error('EP must contain whole nodes and node-local TP groups');
  fraction(o.cpuVectorFraction,'cpuVectorFraction');
  if (![0,1].includes(o.metadataOnDevice)) throw new Error('metadataOnDevice must be 0 or 1');
  positiveInt(o.vectorElementBytes,'vectorElementBytes');
  const lookupsPerToken = geometry.layers * (geometry.ngram - 1) * geometry.heads;
  const vectorBytesPerToken = lookupsPerToken * geometry.width * o.vectorElementBytes;
  const dp = ep / tp, queryCount = tokensPerDP * lookupsPerToken;
  // TP leader contributes the query once; all node owners serve it. The following
  // vector balance is an owner-load assumption, not division by the global EP.
  const vectorBytesPerRank = tokensPerDP / tp * vectorBytesPerToken * o.cpuVectorFraction;
  const idAndOrderBytesPerLeader = queryCount * 8 * 2;
  const metadataBytesPerRank = rounds * geometry.layers * (o.ranksPerNode + 1) * 8 * o.metadataOnDevice;
  const auxiliaryBytesPerRank = idAndOrderBytesPerLeader / tp + metadataBytesPerRank;
  const totalBytesPerRank = vectorBytesPerRank + auxiliaryBytesPerRank;
  return { dp, lookupsPerToken, vectorBytesPerToken, vectorBytesPerRank,
    auxiliaryBytesPerRank, totalBytesPerRank,
    totalBytesPerNode: totalBytesPerRank * o.ranksPerNode,
    totalBytesPerEP: totalBytesPerRank * ep,
    leaderBytes: vectorBytesPerRank + idAndOrderBytesPerLeader + metadataBytesPerRank,
    // Reverse direction, reported separately and never added to H2D.
    d2hBytesPerRank: queryCount / tp * 8 + metadataBytesPerRank * o.ranksPerNode,
  };
}

export async function loadPrefixModels() {
  const models = [...communicationModels.slice(0,5),
    { name:'DS V4.1 Flash', slug:'deepseek-v4.1-flash', profile:'deepseek_v41' }];
  for (const m of models) {
    const raw = await readModelConfig(m.slug), c = raw.text_config || raw;
    m.prefillTP=8; m.decodeTP=['kimi','qwen38'].includes(m.profile)?8:1;
    m.poolAlignment=m.profile==='deepseek_v4'?16384:128;
    m.components=[];
    const add=(name,layers,elements,elementBytes,{extraBytes=0,ratio=1,blockFactor=1,
      window=0,mode='full',tpMode='replicated',heads=1,stateRows=1,kind='cache',note=''}={})=>
      m.components.push({name,layers,elements,elementBytes,extraBytes,ratio,blockFactor,window,mode,tpMode,heads,stateRows,kind,note});
    if(m.profile==='deepseek_v4') {
      const ratios=c.compress_ratios.slice(0,c.num_hidden_layers),n4=ratios.filter(x=>x===4).length,n128=ratios.filter(x=>x===128).length;
      add('SWA KV',c.num_hidden_layers,c.head_dim,2,{mode:'tail',window:c.sliding_window});
      for(const [n,l] of [[4,n4],[128,n128]])add(`C${n} KV`,l,c.head_dim,2,{ratio:n,blockFactor:n});
      add('C4 Indexer',n4,c.index_head_dim,1,{extraBytes:2,ratio:4,blockFactor:4,note:'INT8 + 2 B scale'});
      add('C4 compressor state',n4,4*c.head_dim,4,{mode:'tail',blockFactor:1/16,window:8,kind:'state'});
      add('C128 compressor state',n128,2*c.head_dim,4,{mode:'tail',blockFactor:1/4,window:128,kind:'state'});
      add('C4 indexer state',n4,4*c.index_head_dim,4,{mode:'tail',blockFactor:1/16,window:8,kind:'state'});
    } else if(m.profile==='deepseek_v41') {
      add('SWA KV',c.num_hidden_layers,c.head_dim,2,{mode:'tail',window:c.sliding_window});
      for(const ratio of [2,1]) {
        const count=v41Topology(raw).layers.filter(l=>l.ownsKV&&l.ratio===ratio).length;
        add(`C${ratio} shared KV`,count,c.head_dim,2,{ratio});
        add(`C${ratio} shared Indexer`,count,c.index_head_dim,1,{ratio,extraBytes:2,note:'Unique owner planes; INT8 + 2 B scale'});
      }
      m.note='Cacheable planes only; private ring/history require separate restore.';
    } else if(m.profile==='glm53') {
      const indices=Array.from({length:c.num_hidden_layers},(_,i)=>i).filter(i=>Math.max(i-c.index_skip_topk_offset+1,0)%c.index_topk_freq===0).length;
      add('MLA',c.num_hidden_layers,c.kv_lora_rank+c.qk_rope_head_dim,2);
      add('Indexer',indices,c.index_head_dim,2);
    } else {
      let n,width,state,kernel;
      if(m.profile==='kimi') {
        const k=c.linear_attn_config;
        n=k.kda_layers.length; width=3*k.num_heads*k.head_dim;
        state=k.num_heads*k.head_dim*k.head_dim;kernel=k.short_conv_kernel_size;
        add('MLA',c.num_hidden_layers-n,c.kv_lora_rank+c.qk_rope_head_dim,2);
      } else {
        n=c.layer_types.filter(t=>t==='linear_attention').length;
        width=2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim;
        state=c.linear_num_value_heads*c.linear_key_head_dim*c.linear_value_head_dim;kernel=c.linear_conv_kernel_dim;
        add('GQA K+V',c.num_hidden_layers-n,2*c.head_dim,2,{tpMode:'kv-heads',heads:c.num_key_value_heads});
      }
      add('Conv state',n,width,2,{mode:'endpoint',tpMode:'sharded',stateRows:kernel-1,kind:'state',note:'Whole-TP width / TP; add editable speculative conv slots'});
      add('Recurrent state',n,state,4,{mode:'endpoint',tpMode:'sharded',kind:'state',note:'One aligned endpoint state, not one state per token'});
      m.note='mamba_cache_mode=align; matching P/D TP and state layout.';
    }
    m.note ||= 'BF16 KV; PP=PCP=DCP=1; full-block prefix lookup.';
  }
  return models;
}

export function prefixTransfer(model,{ inputTokens,ep=32,...options }) {
  const o={...HOST_TRANSFER_DEFAULTS,prefillTP:model.prefillTP,decodeTP:model.decodeTP,poolAlignment:model.poolAlignment,...options};
  for(const [k,v] of Object.entries({inputTokens,ep,tp:o.prefillTP,blockSize:o.blockSize,poolAlignment:o.poolAlignment,requests:o.requestsPerDP}))positiveInt(v,k);
  fraction(o.prefixHitFraction,'prefixHitFraction');fraction(o.localRequestFraction,'localRequestFraction');
  if(ep%o.prefillTP)throw new Error('EP must be divisible by P TP');
  if(['kimi','qwen38'].includes(model.profile)&&o.prefillTP!==o.decodeTP)throw new Error('Aligned recurrent-state path requires matching P/D TP');
  const hitTokens=Math.floor(inputTokens*o.prefixHitFraction/o.poolAlignment)*o.poolAlignment;
  const parts=model.components.map(p=>{
    const block=o.blockSize*p.blockFactor;
    if(o.poolAlignment%block)throw new Error('Pool alignment must be a multiple of every component raw-token block');
    let rows=0;
    if(hitTokens>0) {
      if(p.mode==='endpoint')rows=p.stateRows+(p.name==='Conv state'?o.speculativeConvSlots:0);
      else if(p.mode==='tail')rows=Math.min(hitTokens/block,Math.ceil((p.window-1)/block))*block/p.ratio;
      else rows=hitTokens/p.ratio;
    }
    let rowBytes=p.elements*p.elementBytes+p.extraBytes;
    if(p.tpMode==='sharded')rowBytes/=o.prefillTP;
    if(p.tpMode==='kv-heads') {
      if(p.heads%o.prefillTP&&o.prefillTP%p.heads)throw new Error('GQA heads and TP must divide one another');
      rowBytes*=Math.max(1,p.heads/o.prefillTP);
    }
    return {...p,rows,rowBytes,bytesPerRank:p.layers*rows*rowBytes};
  });
  const cacheBytesPerRank=parts.filter(p=>p.kind==='cache').reduce((s,p)=>s+p.bytesPerRank,0);
  const stateBytesPerRank=parts.filter(p=>p.kind==='state').reduce((s,p)=>s+p.bytesPerRank,0);
  const bytesPerRankPerRequest=cacheBytesPerRank+stateBytesPerRank;
  const rankBatchBytes=bytesPerRankPerRequest*o.requestsPerDP*(1-o.localRequestFraction);
  return {inputTokens,ep,tp:o.prefillTP,dp:ep/o.prefillTP,hitTokens,parts,
    cacheBytesPerRank,stateBytesPerRank,bytesPerRankPerRequest,
    requestBytesAllTP:bytesPerRankPerRequest*o.prefillTP,
    rankBatchBytes,dpBatchBytes:rankBatchBytes*o.prefillTP,epBatchBytes:rankBatchBytes*ep};
}

export async function hostTransferReport(options={}) {
  const o={...HOST_TRANSFER_DEFAULTS,...options};
  const models=await loadPrefixModels(), geometry=engramGeometry(await readModelConfig('deepseek-v4.1-flash'));
  return { options:o, geometry,
    engram:o.eps.flatMap(ep=>[
      ...['P step','D step'].map(stage=>{const p=stage==='P step';return {stage,ep,...engramTransfer(geometry,{...o,ep,tp:p?o.prefillTP:o.decodeTP,tokensPerDP:p?o.prefillChunkTokens:o.decodeTokensPerDP})};}),
      ...o.lengths.map(inputTokens=>({stage:'P batch',ep,inputTokens,...engramTransfer(geometry,{...o,ep,tp:o.prefillTP,tokensPerDP:inputTokens*o.requestsPerDP,rounds:Math.ceil(inputTokens*o.requestsPerDP/o.prefillChunkTokens)})})),
    ]),
    prefix:models.flatMap(model=>o.eps.flatMap(ep=>o.lengths.map(inputTokens=>({model:model.name,...prefixTransfer(model,{...o,prefillTP:model.prefillTP,decodeTP:model.decodeTP,ep,inputTokens})})))),
  };
}
