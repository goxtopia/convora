import torch
from torch.nn import functional as F
from delta_model import DeltaModel

class SFTModel(DeltaModel):
    def features(self,ids,valid,memory_enabled=True):
        mask=valid.unsqueeze(1);h=self.embed(ids)*mask;memory=self.engram.memory(ids)
        if not memory_enabled:memory=torch.zeros_like(memory)
        for i,(block,d) in enumerate(zip(self.blocks,self.config.dilations)):
            h=self.finish_block(i,h,block(F.pad(h,(2*d,0))))
            h=self.inject_memory(i,h,memory)*mask
        h=self.delta(h.transpose(1,2).contiguous(),valid).transpose(1,2)
        return self.final_norm(h).transpose(1,2)

    def selected(self,ids,valid,indices):
        features=self.features(ids,valid).reshape(-1,self.config.channels).index_select(0,indices)
        return self.head(features.T.unsqueeze(0)).squeeze(0).T

    def forward(self,ids,valid=None,output_start=0,memory_enabled=True,supervised_mask=None,**kwargs):
        if valid is None:valid=torch.ones_like(ids,dtype=torch.bool)
        features=self.features(ids,valid,memory_enabled)[:,output_start:]
        if supervised_mask is not None:
            selected=features[supervised_mask]
            return self.head(selected.T.unsqueeze(0)).squeeze(0).T
        return self.head(features.transpose(1,2)).transpose(1,2)
