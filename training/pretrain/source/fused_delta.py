"""FP-only fused recurrence and analytical backward. No state quantization.

State and reductions use FP32; projection tensors retain their input dtype.
The stored state is transposed [K,V], matching the deployment reference.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def forward_kernel(Q,K,V,A,B,M,O,S,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr):
    bh=tl.program_id(0);batch=bh//H;head=bh%H
    i=tl.arange(0,D);j=tl.arange(0,D)
    state=tl.full((D,D),0,tl.float32)
    for t in range(T):
        off=((batch*T+t)*H+head)*D
        q=tl.load(Q+off+i).to(tl.float32);k=tl.load(K+off+i).to(tl.float32)
        v=tl.load(V+off+j).to(tl.float32)
        a=tl.load(A+(batch*T+t)*H+head).to(tl.float32)
        b=tl.load(B+(batch*T+t)*H+head).to(tl.float32)
        valid=tl.load(M+batch*T+t)
        old=tl.sum(state*k[:,None],0)
        u=b*(v-a*old)
        new=a*state+k[:,None]*u[None,:]
        state=tl.where(valid,new,state)
        out=tl.where(valid,tl.sum(state*q[:,None],0),0.)
        tl.store(O+off+j,out)
        tl.store(S+((bh*T+t)*D+i[:,None])*D+j[None,:],state)

@tr.jit
def backward_kernel(Q,K,V,A,B,M,S,DO,DQ,DK,DV,DA,DB,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr):
    bh=tl.program_id(0);batch=bh//H;head=bh%H
    i=tl.arange(0,D);j=tl.arange(0,D)
    carry=tl.full((D,D),0,tl.float32)
    for r in range(T):
        t=T-1-r;off=((batch*T+t)*H+head)*D
        q=tl.load(Q+off+i).to(tl.float32);k=tl.load(K+off+i).to(tl.float32)
        v=tl.load(V+off+j).to(tl.float32)
        a=tl.load(A+(batch*T+t)*H+head).to(tl.float32)
        b=tl.load(B+(batch*T+t)*H+head).to(tl.float32)
        valid=tl.load(M+batch*T+t)
        state=tl.load(S+((bh*T+t)*D+i[:,None])*D+j[None,:])
        prev=tl.load(S+((bh*T+t-1)*D+i[:,None])*D+j[None,:],mask=t>0,other=0.)
        dy=tl.where(valid,tl.load(DO+off+j).to(tl.float32),0.)
        g=carry+q[:,None]*dy[None,:]
        old=tl.sum(prev*k[:,None],0);error=v-a*old;u=b*error
        du=tl.sum(g*k[:,None],0);dold=-a*b*du
        dq=tl.sum(state*dy[None,:],1)
        # Autograd casts each branch gradient to its BF16 leaf before adding.
        dk_write=tl.sum(g*u[None,:],1).to(K.dtype.element_ty).to(tl.float32)
        dk_lookup=tl.sum(prev*dold[None,:],1).to(K.dtype.element_ty).to(tl.float32)
        dk=dk_write+dk_lookup
        dv=b*du
        da_decay=tl.sum(tl.sum(g*prev,0),0).to(A.dtype.element_ty).to(tl.float32)
        da_lookup=(-tl.sum(du*b*old,0)).to(A.dtype.element_ty).to(tl.float32)
        da=da_decay+da_lookup
        db=tl.sum(du*error,0)
        next_carry=a*g+k[:,None]*dold[None,:]
        carry=tl.where(valid,next_carry,carry)
        tl.store(DQ+off+i,tl.where(valid,dq,0.));tl.store(DK+off+i,tl.where(valid,dk,0.))
        tl.store(DV+off+j,tl.where(valid,dv,0.))
        tl.store(DA+(batch*T+t)*H+head,tl.where(valid,da,0.))
        tl.store(DB+(batch*T+t)*H+head,tl.where(valid,db,0.))

class FusedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,a,b,valid):
        q,k,v,a,b,valid=[x.contiguous() for x in (q,k,v,a,b,valid)]
        batch,time,heads,dim=q.shape
        out=torch.empty(q.shape,device=q.device,dtype=torch.float32)
        states=torch.empty((batch*heads,time,dim,dim),device=q.device,dtype=torch.float32)
        forward_kernel[(batch*heads,)](q,k,v,a,b,valid,out,states,time,heads,dim,enable_fp_fusion=False)
        ctx.save_for_backward(q,k,v,a,b,valid,states)
        return out
    @staticmethod
    def backward(ctx,grad):
        q,k,v,a,b,valid,states=ctx.saved_tensors
        batch,time,heads,dim=q.shape
        grads=[torch.empty_like(x) for x in (q,k,v,a,b)]
        backward_kernel[(batch*heads,)](q,k,v,a,b,valid,states,grad.contiguous(),*grads,time,heads,dim,enable_fp_fusion=False)
        return (*grads,None)

def fused_recurrence(q,k,v,a,b,valid):
    return FusedDelta.apply(q,k,v,a,b,valid)
