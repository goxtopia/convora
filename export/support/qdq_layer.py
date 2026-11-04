"""Static 4x32 training fake-quant semantics exported as ONNX Q/DQ."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class QDQDelta(nn.Module):
    def __init__(self,sequence):
        super().__init__();self.layer=sequence.layer;self.grids=sequence.grids

    def q(self,name,x,head=0):
        g=self.grids[name];i=head if g.per_head else 0
        return torch.fake_quantize_per_tensor_affine(x,float(g.scale[i]),int(g.zp[i]),-128,127)

    def conv(self,x,module):
        weight=module.weight
        scales=(weight.detach().reshape(weight.shape[0],-1).abs().amax(1)/127).clamp_min(1e-8)
        weight=torch.fake_quantize_per_channel_affine(weight,scales,torch.zeros_like(scales,dtype=torch.int32),0,-128,127)
        return F.conv2d(x,weight,module.bias)

    def sum_key(self,x):
        w=self.layer.sum_key
        w=torch.fake_quantize_per_channel_affine(w,torch.ones(1)/127,torch.zeros(1,dtype=torch.int32),0,-128,127)
        return F.conv2d(x,w)

    def forward(self,h,*states):
        x=self.q("h_in",h);q,k,v=self.q("qkv",self.conv(x,self.layer.qkv)).chunk(3,1)
        reads=[];updated=[]
        for i in range(4):
            query=self.q("query",q[:,i*32:(i+1)*32].clamp(-1,1)/math.sqrt(32),i)
            key=self.q("key",k[:,i*32:(i+1)*32].clamp(-1,1)/math.sqrt(32),i)
            value=self.q("value",v[:,i*32:(i+1)*32].clamp(-4,4).reshape(1,1,1,32),i)
            a=self.q("alpha",.95+.04*F.hardsigmoid(self.q("alpha_linear",self.conv(x,self.layer.alpha[i]),i)),i)
            b=self.q("beta",F.hardsigmoid(self.q("beta_linear",self.conv(x,self.layer.beta[i]),i)),i)
            state=self.q("state_in",states[i],i)
            old=self.q("old_sum",self.sum_key(self.q("old_product",state*key,i)),i)
            error=self.q("value_error",value-self.q("alpha_old",a*old,i),i)
            update=self.q("beta_error",b*error,i)
            new=self.q("state_out",self.q("decayed",a*state,i)+self.q("outer",key*update,i),i)
            read=self.q("read_sum",self.sum_key(self.q("read_product",new*query,i)),i)
            updated.append(new);reads.append(read.reshape(1,32,1,1))
        gate=self.q("gate",F.hardsigmoid(self.q("gate_linear",self.conv(x,self.layer.output_gate))))
        output=self.q("output",self.conv(self.q("mixed",torch.cat(reads,1)*gate),self.layer.output))
        return (h+output,*updated)


class QDQHybrid(nn.Module):
    """Parent PTQ grids collected from train; delta keeps its trained grids."""
    def __init__(self,model,delta):
        super().__init__();self.model=model;self.delta=delta;self.ranges={};self.observe=True;self.quantized=False
        self.register_buffer("head_weight",model.output_weight().detach().clone().unsqueeze(2))

    def q(self,name,x):
        if self.observe:
            lo,hi=float(x.detach().min()),float(x.detach().max());old=self.ranges.get(name,(0.,0.))
            self.ranges[name]=(min(0.,lo,old[0]),max(0.,hi,old[1]))
        if not self.quantized:return x
        low,high=self.ranges[name];scale=max((high-low)/255,1e-8);zp=max(-128,min(127,-128-round(low/scale)))
        return torch.fake_quantize_per_tensor_affine(x,scale,zp,-128,127)

    def conv(self,name,x,weight,bias=None):
        if self.quantized:
            scales=(weight.detach().reshape(weight.shape[0],-1).abs().amax(1)/127).clamp_min(1e-8)
            weight=torch.fake_quantize_per_channel_affine(weight,scales,torch.zeros_like(scales,dtype=torch.int32),0,-128,127)
        return self.q(name,F.conv2d(x,weight,bias))

    def forward(self,x,*args):
        h=self.q("x",x);memory=self.q("memory",args[8]);hidden=[]
        for i,block in enumerate(self.model.blocks):
            state=self.q("state"+str(i),args[i]);taps=self.q("taps"+str(i),torch.cat([state,h],3))
            temporal=self.conv("temporal"+str(i),taps,block.weight.unsqueeze(2),block.bias)
            branch=self.conv("pointwise"+str(i),self.q("relu"+str(i),F.relu(temporal)),self.model.pointwise[i].weight.unsqueeze(2))
            h=self.q("residual"+str(i),h+branch)
            if i==self.model.config.hook:
                m=self.model.engram
                a=self.conv("gwh",h,m.gwh.weight.unsqueeze(2))
                b=self.conv("gwm",memory,m.gwm.weight.unsqueeze(2),m.gwm.bias)
                gate=self.q("engram_gate",F.hardsigmoid(self.q("gate_add",a+b)))
                value=self.conv("gv",memory,(m.gv.weight*m.alpha.reshape(-1,1,1)).unsqueeze(2))
                h=self.q("engram_out",h+self.q("engram_mix",gate*value))
            if i<7:hidden.append(h)
        h,*states=self.delta(h,*args[9:])
        logits=self.conv("logits",self.q("head_input",h),self.head_weight)
        return (logits,*hidden,*states)
