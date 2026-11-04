"""Same epoch windows/labels as baseline, adding all preceding document tokens."""
import numpy as np
import torch
from training.data import TokenCorpus

class FullPrefixCorpus(TokenCorpus):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);assert self.manifest['mode']=='text' and self.block_size==256
        self.max_prefix=512;self.width=768
    def batch_windows(self,windows):
        x=np.zeros((len(windows),self.width),np.int64);valid=np.zeros_like(x,np.bool_);y=np.full((len(windows),self.block_size),-100,np.int64)
        for row,w in enumerate(windows):
            sid=int(np.searchsorted(self.cumulative,w,side='right'));local=int(w)-(int(self.cumulative[sid-1]) if sid else 0)
            tokens,index,cumulative,mask=self.shards[sid];assert mask is None
            doc=int(np.searchsorted(cumulative,local,side='right'));previous=int(cumulative[doc-1]) if doc else 0
            start,end=int(index[doc]),int(index[doc+1]);offset=(local-previous)*self.block_size
            assert offset<=self.max_prefix,'Refuse to crop a longer document prefix'
            body=min(self.block_size,end-start-1-offset)
            x[row,self.max_prefix-offset:self.max_prefix+body]=tokens[start:start+offset+body]
            valid[row,self.max_prefix-offset:self.max_prefix+body]=True
            y[row,:body]=tokens[start+offset+1:start+offset+body+1]
        return torch.from_numpy(x),torch.from_numpy(valid),torch.from_numpy(y)
