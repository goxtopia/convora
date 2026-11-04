"""Weight-shared one/two-pass causal TCN with reusable n-gram memory."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 6400
    channels: int = 256
    dilations: tuple = (1, 2, 4, 8, 16, 32, 32, 32)
    use_engram: bool = True
    num_passes: int = 1
    hook: int = 2  # zero-based: after third residual block
    buckets: tuple = (19997, 20011, 20021, 20023)
    orders: tuple = (2, 2, 3, 3)
    head_dim: int = 64
    tie_embeddings: bool = False
    tied_head_projection: bool = False  # E @ P, with independent input/output channel bases
    temporal_bias: bool = False  # learned threshold before branch ReLU
    block_style: str = "post_relu"  # legacy, or k3_relu_k1 with a linear residual stream
    head_norm: str = "none"  # optional per-token RMSNorm; never normalize over time
    channel_alpha: bool = False
    extra_hooks: tuple = ()  # independent gates, shared lookup tables and memory input
    hash_multipliers: tuple = ((841577377, 1427818425, 1713831177),
                               (1403703963, 542880493, 52321687),
                               (799045305, 480294803, 756998953),
                               (1028136769, 665279961, 1782731651))

    def __post_init__(self):
        if self.num_passes not in (1, 2):
            raise ValueError("num_passes must be 1 or 2")
        for key in ("dilations", "buckets", "orders", "extra_hooks"):
            setattr(self, key, tuple(getattr(self, key)))
        self.hash_multipliers = tuple(tuple(row) for row in self.hash_multipliers)
        if not self.dilations or min(self.dilations) < 1 or not 0 <= self.hook < len(self.dilations):
            raise ValueError("Invalid TCN dilation/hook configuration")
        if not (len(self.buckets) == len(self.orders) == len(self.hash_multipliers)):
            raise ValueError("Memory hash configuration lengths differ")
        if any(n not in (2, 3) for n in self.orders):
            raise ValueError("Only bigram/trigram heads are supported")
        if self.block_style not in ("post_relu", "k3_relu_k1") or self.head_norm not in ("none", "rms"):
            raise ValueError("Invalid block style/head normalization")
        if self.tied_head_projection and not self.tie_embeddings:
            raise ValueError("Tied head projection requires shared embeddings")
        if len(set((self.hook, *self.extra_hooks))) != 1 + len(self.extra_hooks):
            raise ValueError("Engram hooks must be distinct")
        if any(not 0 <= i < len(self.dilations) for i in self.extra_hooks):
            raise ValueError("Engram hook outside TCN")
        if not self.use_engram and self.extra_hooks:
            raise ValueError("Extra Engram hooks require memory")

    @property
    def pass_context(self):
        return 2 * sum(self.dilations)

    @property
    def context(self):
        return self.num_passes * self.pass_context

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


class EngramInjection(nn.Module):
    """A separate gate/value projection; no duplicate memory tables."""
    def __init__(self, config):
        super().__init__()
        self.init_gate(config)

    def init_gate(self, config):
        c = config.channels
        self.gwh = nn.Conv1d(c, c, 1, bias=False)
        self.gwm = nn.Conv1d(c, c, 1, bias=True)
        self.gv = nn.Conv1d(c, c, 1, bias=False)
        self.alpha = nn.Parameter(torch.full((1, c, 1), .25) if config.channel_alpha else torch.tensor(.25))

    def forward(self, h, memory):
        gate = F.hardsigmoid(self.gwh(h) + self.gwm(memory))
        return h + self.alpha * gate * self.gv(memory)


class Engram(EngramInjection):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        if len(config.buckets) * config.head_dim != config.channels:
            raise ValueError("Memory head dimensions must concatenate to the TCN width")
        self.tables = nn.ModuleList(nn.Embedding(n, config.head_dim) for n in config.buckets)
        self.init_gate(config)

    def indices(self, ids):
        previous = (ids, F.pad(ids[:, :-1], (1, 0)),
                    F.pad(ids[:, :-2], (2, 0))[:, :ids.shape[1]])
        result = []
        for n, order, multipliers in zip(self.config.buckets, self.config.orders, self.config.hash_multipliers):
            value = previous[0] * multipliers[0]
            for k in range(1, order):
                value = torch.bitwise_xor(value, previous[k] * multipliers[k])
            result.append(torch.remainder(torch.bitwise_and(value, 0x7fffffff), n))
        return result

    def memory(self, ids):
        return torch.cat([table(key) for table, key in zip(self.tables, self.indices(ids))], dim=-1).transpose(1, 2)


class TiedHead(nn.Module):
    def __init__(self, embedding, projection=False):
        super().__init__()
        # One Parameter object, so AdamW has one state/update for both roles.
        self.weight = embedding.weight
        c = embedding.embedding_dim
        self.proj = nn.Conv1d(c, c, 1, bias=False) if projection else None

    def effective_weight(self):
        if self.proj is None:
            return self.weight.unsqueeze(-1)
        # Column-vector convention: logits = E (P h). Applying conv1d to the
        # rows of E would give E @ P.T instead, and is NOT equivalent.
        return (self.weight @ self.proj.weight[:, :, 0]).unsqueeze(-1)

    def forward(self, h):
        if self.proj is not None:
            h = self.proj(h)
        return F.conv1d(h, self.weight.unsqueeze(-1))


class ChannelRMSNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, h):
        # Also accepts NCHW for static export; channel is always axis 1.
        value = h.float() if h.dtype in (torch.float16, torch.bfloat16) else h
        value = value * torch.rsqrt(value.square().mean(dim=1, keepdim=True) + self.eps)
        shape = (1, -1) + (1,) * (h.ndim - 2)
        return value.to(h.dtype) * self.weight.to(h.dtype).reshape(shape)


class TCNLanguageModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.channels
        self.embedding = nn.Embedding(self.config.vocab_size, c)
        self.blocks = nn.ModuleList(nn.Conv1d(c, c, 3, dilation=d, bias=self.config.temporal_bias)
                                    for d in self.config.dilations)
        self.engram = Engram(self.config) if self.config.use_engram else None
        self.pointwise = nn.ModuleList(nn.Conv1d(c, c, 1, bias=False) for _ in self.blocks) \
            if self.config.block_style == "k3_relu_k1" else nn.ModuleList()
        self.extra_engram = nn.ModuleDict({str(i): EngramInjection(self.config) for i in self.config.extra_hooks})
        self.final_norm = ChannelRMSNorm(c) if self.config.head_norm == "rms" else nn.Identity()
        self.head = TiedHead(self.embedding, self.config.tied_head_projection) if self.config.tie_embeddings else nn.Conv1d(c, self.config.vocab_size, 1, bias=False)
        # Tied output rows keep std=1/sqrt(C); scale only the input lookup to std~1.
        self.embedding_scale = math.sqrt(c) if self.config.tie_embeddings else 1.
        nn.init.normal_(self.embedding.weight, std=c**-.5 if self.config.tie_embeddings else 1.)
        for block in self.blocks:
            nn.init.normal_(block.weight, std=.2 / (3*c)**.5)
            if block.bias is not None:
                nn.init.zeros_(block.bias)
        for block in self.pointwise:
            nn.init.normal_(block.weight, std=c**-.5)
        if not self.config.tie_embeddings:
            nn.init.normal_(self.head.weight, std=c**-.5)
        elif self.head.proj is not None:
            # Identity initialization would retain the current-token logit spike.
            nn.init.normal_(self.head.proj.weight, std=c**-.5)
        if self.engram is not None:
            for table in self.engram.tables:
                nn.init.normal_(table.weight, std=1.)
            for gate in (self.engram, *self.extra_engram.values()):
                nn.init.normal_(gate.gwh.weight, std=.2*c**-.5)
                nn.init.normal_(gate.gwm.weight, std=.2*c**-.5)
                nn.init.zeros_(gate.gwm.bias)
                nn.init.normal_(gate.gv.weight, std=c**-.5)

    def embed(self, ids):
        return self.embedding(ids).transpose(1, 2) * self.embedding_scale

    def output_weight(self):
        return self.head.effective_weight() if self.config.tie_embeddings else self.head.weight

    def finish_block(self, i, h, temporal):
        if self.config.block_style == "post_relu":
            return F.relu(h + temporal)
        return h + self.pointwise[i](F.relu(temporal))

    def inject_memory(self, i, h, memory):
        if self.engram is not None and i == self.config.hook:
            h = self.engram(h, memory)
        if str(i) in self.extra_engram:
            h = self.extra_engram[str(i)](h, memory)
        return h

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if self.config.tie_embeddings:
            a, b = state_dict.get("embedding.weight"), state_dict.get("head.weight")
            if a is not None and b is not None and not torch.equal(a, b):
                raise ValueError("Tied checkpoint contains different embedding/head weights; explicit migration is required")
            if kwargs.get("assign", False):
                raise ValueError("assign=True can break tied Parameter identity; use the default copy load")
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def forward(self, ids, valid=None, output_start=0, memory_enabled=True, supervised_mask=None):
        # Mask every block, not just the embedding: otherwise padded activations enter later taps.
        mask = valid.unsqueeze(1) if valid is not None else None
        h = self.embed(ids)
        if mask is not None:
            h = h * mask
        memory = self.engram.memory(ids) if self.engram is not None else None
        if memory is not None and not memory_enabled:
            memory = torch.zeros_like(memory)
        # Each pass has its own temporal activations, but all passes share learned weights.
        # Memory lookup happens once; gradients flow through both passes without detach().
        for _ in range(self.config.num_passes):
            for i, (block, dilation) in enumerate(zip(self.blocks, self.config.dilations)):
                h = self.finish_block(i, h, block(F.pad(h, (2*dilation, 0))))
                h = self.inject_memory(i, h, memory)
                if mask is not None:
                    h = h * mask
        # The expensive vocabulary projection is needed only at supervised positions.
        features = self.final_norm(h[:, :, output_start:])
        if supervised_mask is not None:
            if supervised_mask.shape != (features.shape[0], features.shape[2]):
                raise ValueError("Supervision mask shape differs from output positions")
            selected = features.transpose(1, 2)[supervised_mask]
            return self.head(selected.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        return self.head(features).transpose(1, 2)

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        memory = sum(p.numel() for table in self.engram.tables for p in table.parameters()) if self.engram else 0
        gate = (sum(p.numel() for p in self.engram.parameters()) - memory +
                sum(p.numel() for p in self.extra_engram.parameters())) if self.engram else 0
        return {"base": total-memory-gate, "memory": memory, "gate": gate, "total": total}
