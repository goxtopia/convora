"""Deployment-oriented gated delta rule, state transposed to [1,K,1,V].

Per-head state T=S^T. Read: v_old = sum_K(T*k). Write:
T_new = alpha*T + k*[beta*(v-alpha*v_old)]. Read query from T_new.
Dynamic products become Mul; key-axis sums are fixed 1x1 convolutions.
Bounded keys/queries (norm <= 1), HardSigmoid gates and no new shortconv/
output norm are deliberate adaptations, NOT the full paper architecture.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class DeltaLayer(nn.Module):
    def __init__(self, channels=256, heads=4, dimension=64, normalize="bounded"):
        super().__init__()
        self.channels, self.heads, self.dimension, self.normalize = channels, heads, dimension, normalize
        width = heads*dimension
        self.qkv = nn.Conv2d(channels, 3*width, 1, bias=False)
        self.alpha = nn.ModuleList(nn.Conv2d(channels, 1, 1) for _ in range(heads))
        self.beta = nn.ModuleList(nn.Conv2d(channels, 1, 1) for _ in range(heads))
        self.output_gate = nn.Conv2d(channels, width, 1)
        self.output = nn.Conv2d(width, channels, 1, bias=False)
        self.register_buffer("sum_key", torch.ones(1, dimension, 1, 1))
        nn.init.normal_(self.qkv.weight, std=.15/math.sqrt(channels))
        nn.init.normal_(self.output.weight, std=.25/math.sqrt(width))
        for layer in [*self.alpha, *self.beta, self.output_gate]:
            nn.init.normal_(layer.weight, std=.1/math.sqrt(channels))
            nn.init.zeros_(layer.bias)

    def key_query(self, value):
        if self.normalize == "l2":
            return F.normalize(value, p=2, dim=1, eps=1e-6)
        return F.hardtanh(value, -1., 1.)/math.sqrt(self.dimension)

    def forward(self, h, *states):
        d = self.dimension
        q, k, v = self.qkv(h).chunk(3, dim=1)
        values, updated = [], []
        for i in range(self.heads):
            query = self.key_query(q[:, i*d:(i+1)*d])
            key = self.key_query(k[:, i*d:(i+1)*d])
            value = F.hardtanh(v[:, i*d:(i+1)*d], -4., 4.).reshape(1, 1, 1, d)
            a = .95+.04*F.hardsigmoid(self.alpha[i](h))
            b = F.hardsigmoid(self.beta[i](h))
            old = F.conv2d(states[i]*key, self.sum_key)
            next_state = a*states[i] + key*(b*(value-a*old))
            read = F.conv2d(next_state*query, self.sum_key)
            values.append(read.reshape(1, d, 1, 1)); updated.append(next_state)
        out = torch.cat(values, dim=1)*F.hardsigmoid(self.output_gate(h))
        return (h+self.output(out), *updated)

    def reference(self, h, states):
        """Independent matrix-multiply formulation for numerical verification."""
        d = self.dimension
        q, k, v = self.qkv(h).chunk(3, dim=1)
        values, updated = [], []
        for i in range(self.heads):
            query = self.key_query(q[:, i*d:(i+1)*d]).reshape(d)
            key = self.key_query(k[:, i*d:(i+1)*d]).reshape(d)
            value = F.hardtanh(v[:, i*d:(i+1)*d], -4., 4.).reshape(d)
            a = (.95+.04*F.hardsigmoid(self.alpha[i](h))).reshape(())
            b = F.hardsigmoid(self.beta[i](h)).reshape(())
            state = states[i].reshape(d, d)
            old = state.T @ key
            next_state = a*state+torch.outer(key, b*(value-a*old))
            values.append((next_state.T @ query).reshape(1, d, 1, 1))
            updated.append(next_state.reshape(1, d, 1, d))
        out = torch.cat(values, 1)*F.hardsigmoid(self.output_gate(h))
        return (h+self.output(out), *updated)
