"""Reference streaming decoder: shared weights, separate temporal caches per pass."""
import torch
from torch.nn import functional as F


class CachedDecoder:
    def __init__(self, model, batch_size=1, memory_enabled=True):
        self.model = model
        self.batch_size = batch_size
        self.memory_enabled = memory_enabled
        self.reset()

    def reset(self):
        cfg = self.model.config
        weight = self.model.embedding.weight
        self.rings = [[weight.new_zeros(self.batch_size, cfg.channels, 2*d) for d in cfg.dilations]
                      for _ in range(cfg.num_passes)]
        self.pointers = [[0]*len(cfg.dilations) for _ in range(cfg.num_passes)]
        self.commits = [[0]*len(cfg.dilations) for _ in range(cfg.num_passes)]
        self.previous_ids = torch.zeros(self.batch_size, 2, dtype=torch.long, device=weight.device)
        self.tokens_seen = 0

    @torch.no_grad()
    def step(self, token):
        cfg = self.model.config
        token = torch.as_tensor(token, dtype=torch.long, device=self.previous_ids.device).reshape(self.batch_size, 1)
        h = self.model.embed(token)
        memory = None
        if self.model.engram is not None:
            history = torch.cat((self.previous_ids, token), dim=1)
            memory = self.model.engram.memory(history)[:, :, -1:]
            if not self.memory_enabled:
                memory = torch.zeros_like(memory)
        for r in range(cfg.num_passes):
            for i, (block, d) in enumerate(zip(self.model.blocks, cfg.dilations)):
                ring, pointer = self.rings[r][i], self.pointers[r][i]
                incoming = h
                taps = torch.cat((ring[:, :, pointer:pointer+1],
                                  ring[:, :, (pointer+d) % (2*d):(pointer+d) % (2*d)+1], h), dim=2)
                h = self.model.finish_block(i, h, F.conv1d(taps, block.weight, block.bias))
                h = self.model.inject_memory(i, h, memory)
                ring[:, :, pointer:pointer+1].copy_(incoming)
                self.pointers[r][i] = (pointer+1) % (2*d)
                self.commits[r][i] += 1
        self.previous_ids = torch.cat((self.previous_ids[:, 1:], token), dim=1)
        self.tokens_seen += 1
        return self.model.head(self.model.final_norm(h)).squeeze(2)
