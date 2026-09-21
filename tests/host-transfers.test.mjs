import test from 'node:test';
import assert from 'node:assert/strict';
import { loadPrefixModels, engramGeometry, engramTransfer, prefixTransfer, hostTransferReport } from '../lib/host-transfers.mjs';
import { readModelConfig } from '../lib/model-catalog.mjs';

test('FP8 table offload returns 48 BF16 vectors per token; H2D is not table storage bytes',async()=>{
  const g=engramGeometry(await readModelConfig('deepseek-v4.1-flash'));
  const p=engramTransfer(g,{ep:32,tp:8,tokensPerDP:16384});
  assert.equal(p.lookupsPerToken,48);assert.equal(p.vectorBytesPerToken,24576);
  assert.equal(p.vectorBytesPerRank,48*2**20);
  assert.equal(p.totalBytesPerEP,16384*4*(24576+768)+32*144);
  assert.equal(p.leaderBytes-p.vectorBytesPerRank,16384*768+144);
  const d=engramTransfer(g,{ep:32,tp:1,tokensPerDP:128});
  assert.equal(d.vectorBytesPerRank,3*2**20);
  assert.equal(engramTransfer(g,{ep:256,tp:8,tokensPerDP:16384}).totalBytesPerEP,p.totalBytesPerEP*8);
  assert.equal(engramTransfer(g,{ep:32,tp:8,tokensPerDP:16384,cpuVectorFraction:0}).vectorBytesPerRank,0);
  assert.throws(()=>engramTransfer(g,{ep:32,tp:16,tokensPerDP:1}),/node-local/);
});

test('pool prefix uses full blocks, minimal SWA suffix, and one recurrent endpoint',async()=>{
  const models=await loadPrefixModels();
  for(const model of models){
    const a=prefixTransfer(model,{inputTokens:16384}),b=prefixTransfer(model,{inputTokens:16384,ep:256});
    assert.equal(b.epBatchBytes,a.epBatchBytes*8);assert.equal(b.rankBatchBytes,a.rankBatchBytes);
    assert.equal(prefixTransfer(model,{inputTokens:16384,prefixHitFraction:0}).epBatchBytes,0);
    assert.equal(prefixTransfer(model,{inputTokens:16384,localRequestFraction:1}).epBatchBytes,0);
    assert.equal(prefixTransfer(model,{inputTokens:model.poolAlignment-1}).hitTokens,0);
    if(['kimi','qwen38'].includes(model.profile)){
      assert.equal(a.parts.find(x=>x.name==='Recurrent state').rows,1);
      assert.equal(prefixTransfer(model,{inputTokens:1048576}).stateBytesPerRank,a.stateBytesPerRank);
    }
    if(model.profile==='deepseek_v41'){
      assert.equal(a.parts.find(x=>x.name==='SWA KV').rows,128);
      assert.equal(a.stateBytesPerRank,0);
      assert.equal(a.parts.filter(x=>x.name.includes('shared KV')).reduce((s,x)=>s+x.layers,0),4);
      // Hand calculation: 40 SWA rows; 3 C2 + 1 C1 KV/index planes, all TP8 replicated.
      assert.equal(a.requestBytesAllTP,(40*128*1024+(3*8192+16384)*(1024+130))*8);
    }
    if(model.profile==='qwen38'){
      assert.equal(a.parts.find(x=>x.name==='GQA K+V').rowBytes,2*256*2);
      const tp4=prefixTransfer(model,{inputTokens:16384,prefillTP:4,decodeTP:4});
      // Four KV heads: TP8 duplicates each head twice, TP4 stores one copy.
      assert.equal(tp4.cacheBytesPerRank*4,a.cacheBytesPerRank*8/2);
      assert.equal(tp4.stateBytesPerRank*4,a.stateBytesPerRank*8);
    }
  }
});

test('export keeps each model TP separate from Engram D TP1',async()=>{
  const report=await hostTransferReport();
  assert.equal(report.prefix.length,36);assert.equal(report.engram.length,10);
  assert.equal(report.prefix.find(x=>x.model==='KimiK3').tp,8);
});
