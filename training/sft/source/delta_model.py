"""Fresh TCN + Engram + one FP Delta Lite layer; no QAT or inherited weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from training.model import TCNLanguageModel as BaseTCN
from delta_layer import DeltaLayer
from qat_delta import recurrent_step

class FPDelta(nn.Module):
    def __init__(self):
        super().__init__();self.layer=DeltaLayer(dimension=32,normalize='bounded');self.step_function=recurrent_step;self.backend='reference'
    def compile_step(self):self.step_function=torch.compile(recurrent_step,fullgraph=True)
    def linear(self,x,module):return F.linear(x,module.weight.reshape(module.weight.shape[0],-1),module.bias)
    def forward(self,features,valid):
        q,k,v=[x.reshape(*features.shape[:2],4,32) for x in self.linear(features,self.layer.qkv).chunk(3,-1)]
        q=q.clamp(-1,1)/math.sqrt(32);k=k.clamp(-1,1)/math.sqrt(32);v=v.clamp(-4,4)
        a=.95+.04*F.hardsigmoid(torch.stack([self.linear(features,m) for m in self.layer.alpha],2))
        b=F.hardsigmoid(torch.stack([self.linear(features,m) for m in self.layer.beta],2))
        gate=F.hardsigmoid(self.linear(features,self.layer.output_gate))
        if self.backend in ('fused','chunk'):
            if self.backend=='fused':from fused_delta import fused_recurrence as recurrence
            else:from chunk_delta import chunk_recurrence as recurrence
            read=recurrence(q,k,v,a,b,valid).reshape(features.shape[0],features.shape[1],128)
        else:
            projected=[x.transpose(0,1).contiguous() for x in (q,k,v,a,b)]
            state=features.new_zeros(features.shape[0],4,32,32);reads=[]
            for t in range(features.shape[1]):
                read,state=self.step_function(state,*[x[t] for x in projected],valid[:,t].contiguous(),None,None,False)
                reads.append(read.reshape(features.shape[0],128))
            read=torch.stack(reads,1)
        residual=self.linear(read*gate,self.layer.output)
        return (features+residual)*valid.unsqueeze(-1)

class DeltaModel(BaseTCN):
    def __init__(self,config):
        super().__init__(config);assert config.num_passes==1
        self.delta=FPDelta()
    def parameter_counts(self):
        result=super().parameter_counts();count=sum(p.numel() for p in self.delta.parameters())
        result['base']-=count;result['delta']=count;return result
    def forward(self,ids,valid=None,output_start=0,memory_enabled=True,delta_enabled=True,**kwargs):
        if valid is None:valid=torch.ones_like(ids,dtype=torch.bool)
        mask=valid.unsqueeze(1);h=self.embed(ids)*mask;memory=self.engram.memory(ids)
        if not memory_enabled:memory=torch.zeros_like(memory)
        for i,(block,d) in enumerate(zip(self.blocks,self.config.dilations)):
            h=self.finish_block(i,h,block(F.pad(h,(2*d,0))))
            h=self.inject_memory(i,h,memory)*mask
        if delta_enabled and self.delta.backend!='disabled':h=self.delta(h.transpose(1,2).contiguous(),valid).transpose(1,2)
        return self.head(self.final_norm(h[:,:,output_start:])).transpose(1,2)
