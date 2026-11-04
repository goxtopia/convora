import math
import torch
from torch.nn import functional as F
from sft_data import SPECS

def loss_sum(logits,labels):
    return F.cross_entropy(logits.float(),labels,ignore_index=-100,reduction='sum')

class SparseGraph:
    def __init__(self,model,width,enabled=True,stream=None,pool=None):
        self.model=model;self.width=width;self.micro,self.block,self.capacity=SPECS[width]
        self.params=list(model.parameters());self.stream=stream or torch.cuda.Stream();self.stream.wait_stream(torch.cuda.current_stream())
        self.loss_fn=torch.compile(loss_sum,fullgraph=True);self.replays=0;self.eager=0
        with torch.cuda.stream(self.stream):
            for p in self.params:
                if p.grad is None:p.grad=torch.zeros_like(p)
                else:p.grad.zero_()
            self.ids=torch.zeros((self.micro,width),device='cuda',dtype=torch.long);self.valid=torch.ones_like(self.ids,dtype=torch.bool)
            self.indices=torch.zeros(self.capacity,device='cuda',dtype=torch.long)
            self.targets=torch.zeros(self.capacity,device='cuda',dtype=torch.long)
            self.denominator=torch.full((),float(self.capacity),device='cuda');self.loss=torch.empty((),device='cuda')
            if enabled:
                for _ in range(2):self.zero();self._backward()
                self.zero()
        self.stream.synchronize();self.graph=None
        if enabled:
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=self.stream,pool=pool):self._backward()
        self.stream.synchronize();torch.cuda.current_stream().wait_stream(self.stream)
        self.pointers=[p.grad.data_ptr() for p in self.params]
    def zero(self):
        for p in self.params:p.grad.zero_()
    def _backward(self):
        with torch.autocast('cuda',enabled=False):
            logits=self.model.selected(self.ids,self.valid,self.indices)
            total=self.loss_fn(logits,self.targets)
        (total/self.denominator).backward();self.loss.copy_(total.detach())
    def backward(self,batch,total_targets):
        x,valid,y,indices,targets=batch;assert len(x)<=self.micro and len(targets)<=self.capacity and total_targets>0
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.ids.zero_();self.valid.zero_();self.targets.fill_(-100);self.indices.zero_()
            self.ids[:len(x)].copy_(x);self.valid[:len(x)].copy_(valid)
            self.targets[:len(targets)].copy_(targets);self.indices[:len(indices)].copy_(indices);self.denominator.fill_(float(total_targets))
            if self.graph is None:self._backward();self.eager+=1
            else:self.graph.replay();self.replays+=1
            value=float(self.loss)
        torch.cuda.current_stream().wait_stream(self.stream)
        assert self.pointers==[p.grad.data_ptr() for p in self.params]
        if not math.isfinite(value):raise FloatingPointError('nonfinite sparse CE')
        return value

class Engine:
    def __init__(self,model,enabled=True):
        self.graphs={};self.model=model;stream=torch.cuda.Stream()
        # Separate pools permit arbitrary group execution order. Combined
        # allocation is measured under the same 12GiB cap during preflight.
        for width in SPECS:
            self.graphs[width]=SparseGraph(model,width,enabled=enabled,stream=stream)
    def backward(self,packed,total_targets):
        value=0.
        for width,batch in packed:
            value+=self.graphs[width].backward(tuple(t.cuda() for t in batch),total_targets)
        return value
