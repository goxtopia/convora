"""BF16 microbatch forward/backward graph, exact whole-update target normalization."""
import torch
from torch.nn import functional as F

def loss_sum(logits,labels):
    return F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='sum')

class MicroGraph:
    def __init__(self,model,micro,width=768,block=256,enabled=True,compiled_loss=False,pool=None,stream=None):
        self.model=model;self.params=list(model.parameters());self.micro=micro;self.width=width;self.block=block;self.enabled=enabled
        self.device=self.params[0].device;self.stream=stream if stream is not None else torch.cuda.Stream(device=self.device)
        self.loss_function=torch.compile(loss_sum,fullgraph=True) if compiled_loss else loss_sum
        self.stream.wait_stream(torch.cuda.current_stream(self.device));self.replays=0;self.eager_calls=0
        with torch.cuda.stream(self.stream):
            for p in self.params:
                if p.grad is None:p.grad=torch.zeros_like(p)
                else:p.grad.zero_()
            self.ids=torch.zeros(micro,width,dtype=torch.long,device=self.device);self.valid=torch.ones_like(self.ids,dtype=torch.bool)
            self.labels=torch.zeros(micro,block,dtype=torch.long,device=self.device);self.denominator=torch.full((),float(micro*block),device=self.device)
            self.loss=torch.empty((),device=self.device)
            if enabled:
                for _ in range(2):self.zero();self._backward()
                self.zero()
        self.stream.synchronize();self.graph=None
        if enabled:
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=self.stream,pool=pool):self._backward()
        self.pointers=[p.grad.data_ptr() for p in self.params];self.synchronize()
    def zero(self):
        for p in self.params:p.grad.zero_()
    def _backward(self):
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            logits=self.model(self.ids,self.valid,self.width-self.block)
            total=self.loss_function(logits,self.labels)
        (total/self.denominator).backward();self.loss.copy_(total.detach())
    def backward(self,x,valid,y,update_tokens):
        assert x.shape[1]==self.width and y.shape[1]==self.block and len(x)<=self.micro and update_tokens>0
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            if len(x)<self.micro:self.ids.zero_();self.valid.zero_();self.labels.fill_(-100)
            self.ids[:len(x)].copy_(x);self.valid[:len(x)].copy_(valid);self.labels[:len(x)].copy_(y);self.denominator.fill_(float(update_tokens))
            if self.graph is not None:self.graph.replay();self.replays+=1
            else:self._backward();self.eager_calls+=1
            value=float(self.loss)
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        assert [p.grad.data_ptr() for p in self.params]==self.pointers
        if not __import__('math').isfinite(value):raise FloatingPointError('Nonfinite pretraining loss')
        return value
    def synchronize(self):
        self.stream.synchronize();torch.cuda.current_stream(self.device).wait_stream(self.stream)
