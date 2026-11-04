"""4x32 gated-delta training reference with explicit fake-quantized recurrence.

Known state input/output affine grids are copied from the previous board probe.
Other activation grids are calibrated in the training framework. This does NOT
claim bit-exact simulation of RKNN fused operators or a deployable QAT export.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

HEADS, DIM = 4, 32
STATE_IN_SCALE = [.0191876777, .0189029928, .0199464895, .0183004048]
STATE_IN_ZP = [9, 3, -9, -4]
STATE_OUT_SCALE = [.0210212916, .0186409559, .0207455587, .0191941001]
STATE_OUT_ZP = [13, 4, -10, -3]
STEP_NAMES = ("state_in", "old_product", "old_sum", "alpha_old", "value_error",
              "beta_error", "decayed", "outer", "state_out", "read_product", "read_sum")


def fq(x, scale, zp):
    """Affine int8 simulation with saturation-aware straight-through gradient."""
    integer = torch.round(x/scale)+zp
    result = (integer.clamp(-128, 127)-zp)*scale
    inside = ((integer >= -128) & (integer <= 127)).to(x.dtype).detach()
    return result.detach()+(x-x.detach())*inside


def recurrent_step(state, query, key, value, alpha, beta, valid, scales, zeros, quant):
    def quantize(x, index):
        if not quant: return x
        shape = (1, HEADS)+(1,)*(x.ndim-2)
        return fq(x, scales[index].reshape(shape), zeros[index].reshape(shape))
    incoming = quantize(state, 0)
    old = quantize(quantize(incoming*key.unsqueeze(-1), 1).sum(-2), 2)
    error = quantize(value-quantize(alpha*old, 3), 4)
    update = quantize(beta*error, 5)
    decayed = quantize(alpha.unsqueeze(-1)*incoming, 6)
    outer = quantize(key.unsqueeze(-1)*update.unsqueeze(-2), 7)
    outgoing = quantize(decayed+outer, 8)
    read = quantize(quantize(outgoing*query.unsqueeze(-1), 9).sum(-2), 10)
    next_state = torch.where(valid[:, None, None, None], outgoing, state)
    read = torch.where(valid[:, None, None], read, torch.zeros_like(read))
    return read, next_state


class Grid(nn.Module):
    def __init__(self, per_head=False, scale=None, zp=None):
        super().__init__()
        count = HEADS if per_head else 1
        self.per_head, self.fixed = per_head, scale is not None
        self.register_buffer("scale", torch.tensor(scale if scale is not None else [1.]*count, dtype=torch.float32))
        self.register_buffer("zp", torch.tensor(zp if zp is not None else [0.]*count, dtype=torch.float32))
        self.register_buffer("minimum", torch.full((count,), float("inf")))
        self.register_buffer("maximum", torch.full((count,), -float("inf")))

    def observe(self, x, head_axis):
        if not self.fixed:
            with torch.no_grad():
                if self.per_head:
                    axes = tuple(i for i in range(x.ndim) if i != head_axis)
                    low, high = x.detach().amin(axes), x.detach().amax(axes)
                else:
                    low, high = x.detach().amin().reshape(1), x.detach().amax().reshape(1)
                self.minimum.copy_(torch.minimum(self.minimum, low)); self.maximum.copy_(torch.maximum(self.maximum, high))
        return x

    def freeze(self):
        if not self.fixed:
            assert bool(torch.isfinite(self.minimum).all()) and bool(torch.isfinite(self.maximum).all())
            low = torch.minimum(self.minimum, torch.zeros_like(self.minimum))
            high = torch.maximum(self.maximum, torch.zeros_like(self.maximum))
            self.scale.copy_(((high-low)/255).clamp_min(1e-8))
            self.zp.copy_((-128-torch.round(low/self.scale)).clamp(-128, 127))

    def apply(self, x, mode, head_axis=1):
        if mode == "observe": return self.observe(x, head_axis)
        if mode != "qat": return x
        shape = [1]*x.ndim
        if self.per_head: shape[head_axis] = HEADS
        return fq(x, self.scale.reshape(shape), self.zp.reshape(shape))


class SequenceDelta(nn.Module):
    def __init__(self, layer):
        super().__init__()
        assert layer.dimension == DIM and layer.heads == HEADS and layer.normalize == "bounded"
        self.layer = layer
        shared = ["h_in", "qkv", "gate_linear", "gate", "mixed", "output"]
        per_head = ["query", "key", "value", "alpha_linear", "beta_linear", "alpha", "beta", *STEP_NAMES]
        self.grids = nn.ModuleDict({name: Grid(name in per_head) for name in shared+per_head})
        self.grids["state_in"] = Grid(True, STATE_IN_SCALE, STATE_IN_ZP)
        self.grids["state_out"] = Grid(True, STATE_OUT_SCALE, STATE_OUT_ZP)
        self.step_function = recurrent_step

    def use_compiled_step(self):
        self.step_function = torch.compile(recurrent_step, fullgraph=True, mode="default")

    def weight(self, value, mode):
        matrix = value.reshape(value.shape[0], -1)
        if mode == "qat":
            scales = (matrix.detach().abs().amax(1, keepdim=True)/127).clamp_min(1e-8)
            matrix = fq(matrix, scales, torch.zeros_like(scales))
        return matrix

    def linear(self, value, module, mode):
        return F.linear(value, self.weight(module.weight, mode), module.bias)

    def project(self, features, mode):
        grid = lambda name,x: self.grids[name].apply(x, mode, 2)
        h = grid("h_in", features)
        qkv = grid("qkv", self.linear(h, self.layer.qkv, mode))
        q,k,v = [x.reshape(*features.shape[:2], HEADS, DIM) for x in qkv.chunk(3, -1)]
        q = grid("query", q.clamp(-1, 1)/math.sqrt(DIM))
        k = grid("key", k.clamp(-1, 1)/math.sqrt(DIM))
        v = grid("value", v.clamp(-4, 4))
        alpha_linear = torch.stack([self.linear(h, module, mode) for module in self.layer.alpha], dim=2)
        beta_linear = torch.stack([self.linear(h, module, mode) for module in self.layer.beta], dim=2)
        a = grid("alpha", .95+.04*F.hardsigmoid(grid("alpha_linear", alpha_linear)))
        b = grid("beta", F.hardsigmoid(grid("beta_linear", beta_linear)))
        gate = grid("gate", F.hardsigmoid(grid("gate_linear", self.linear(h, self.layer.output_gate, mode))))
        return [x.transpose(0, 1).contiguous() for x in (q,k,v,a,b)], gate

    def observed_step(self, state, q, k, v, a, b, valid):
        g = lambda name,x:self.grids[name].apply(x,"observe",1)
        incoming = g("state_in",state)
        old = g("old_sum",g("old_product",incoming*k.unsqueeze(-1)).sum(-2))
        error = g("value_error",v-g("alpha_old",a*old))
        update = g("beta_error",b*error)
        outgoing = g("state_out",g("decayed",a.unsqueeze(-1)*incoming)+g("outer",k.unsqueeze(-1)*update.unsqueeze(-2)))
        read = g("read_sum",g("read_product",outgoing*q.unsqueeze(-1)).sum(-2))
        return torch.where(valid[:,None,None],read,torch.zeros_like(read)), torch.where(valid[:,None,None,None],outgoing,state)

    def forward(self, features, valid, mode="fp", return_states=False, residual_enabled=True):
        projected, gate = self.project(features, mode)
        state = features.new_zeros(features.shape[0], HEADS, DIM, DIM)
        scales = torch.stack([self.grids[name].scale for name in STEP_NAMES])
        zeros = torch.stack([self.grids[name].zp for name in STEP_NAMES])
        reads, states = [], []
        for t in range(features.shape[1]):
            args = [state, *[value[t] for value in projected], valid[:,t].contiguous()]
            if mode == "observe": read,state = self.observed_step(*args)
            else: read,state = self.step_function(*args, scales, zeros, mode == "qat")
            reads.append(read.reshape(features.shape[0], HEADS*DIM))
            if return_states: states.append(state)
        read = torch.stack(reads, dim=1)
        mixed = self.grids["mixed"].apply(read*gate, mode, 2)
        residual = self.grids["output"].apply(self.linear(mixed,self.layer.output,mode), mode, 2)
        output = features+residual if residual_enabled else features
        output = output*valid.unsqueeze(-1)
        return (output, torch.stack(states,1)) if return_states else output

    def freeze_grids(self):
        for grid in self.grids.values(): grid.freeze()


@torch.no_grad()
def trunk_features(parent, ids, valid):
    mask = valid.unsqueeze(1)
    h = parent.embed(ids)*mask
    memory = parent.engram.memory(ids)
    for i,(block,dilation) in enumerate(zip(parent.blocks,parent.config.dilations)):
        h = parent.finish_block(i,h,block(F.pad(h,(2*dilation,0))))
        h = parent.inject_memory(i,h,memory)*mask
    return h.transpose(1,2).contiguous()
