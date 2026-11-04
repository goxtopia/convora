"""Memory-mapped, document-isolated chunks with deterministic random-access sampling."""
import json
from pathlib import Path

import numpy as np
import torch


class TokenCorpus:
    def __init__(self, manifest_path, split, block_size, context):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        self.block_size, self.context = block_size, context
        self.shards, chunk_counts = [], []
        for shard in self.manifest["shards"]:
            info = shard["splits"][split]
            if not info["documents"]:
                continue
            for filename, expected_size in info["files"].items():
                if (self.path.parent / filename).stat().st_size != expected_size:
                    raise ValueError(f"Dataset file size mismatch: {filename}")
            def path(suffix):
                return self.path.parent / next(n for n in info["files"] if n.endswith(suffix))
            tokens = np.memmap(path(".bin"), dtype="<u2", mode="r")
            index = np.memmap(path(".idx"), dtype="<u8", mode="r")
            if len(index) != info["documents"]+1 or index[0] != 0 or index[-1] != len(tokens):
                raise ValueError("Invalid document index")
            lengths = np.diff(index).astype(np.int64)
            if (lengths < 2).any():
                raise ValueError("Documents must contain at least two tokens")
            windows = (lengths-1 + block_size-1)//block_size
            cumulative = np.cumsum(windows)
            loss = np.memmap(path(".loss"), dtype="u1", mode="r") if self.manifest["mode"] == "chat" else None
            if loss is not None and len(loss) != len(tokens):
                raise ValueError("Invalid assistant loss mask")
            self.shards.append((tokens, index, cumulative, loss))
            chunk_counts.append(int(cumulative[-1]))
        if not self.shards:
            raise ValueError(f"No documents in {split}; use more data/a larger validation fraction")
        self.cumulative = np.cumsum(chunk_counts)
        self.windows = int(self.cumulative[-1])

    def batch(self, batch_size, seed):
        """Seed is derived from the saved microbatch cursor; no hidden worker RNG state."""
        rng = np.random.default_rng(seed)
        width = self.context + self.block_size
        x = np.zeros((batch_size, width), dtype=np.int64)
        valid = np.zeros((batch_size, width), dtype=np.bool_)
        y = np.full((batch_size, self.block_size), -100, dtype=np.int64)
        for row in range(batch_size):
            for attempt in range(10000):
                window = int(rng.integers(self.windows))
                sid = int(np.searchsorted(self.cumulative, window, side="right"))
                local = window - (int(self.cumulative[sid-1]) if sid else 0)
                tokens, index, cumulative, mask = self.shards[sid]
                doc = int(np.searchsorted(cumulative, local, side="right"))
                chunk = local - (int(cumulative[doc-1]) if doc else 0)
                start, end = int(index[doc]), int(index[doc+1])
                offset = chunk * self.block_size
                body = min(self.block_size, end-start-1-offset)
                labels = np.array(tokens[start+offset+1:start+offset+body+1], dtype=np.int64)
                if mask is not None:
                    labels[np.asarray(mask[start+offset+1:start+offset+body+1]) == 0] = -100
                if (labels != -100).any():
                    break
            else:
                raise ValueError("Unable to sample any supervised tokens")
            prefix = min(self.context, offset)
            dest = self.context-prefix
            x[row, dest:self.context+body] = tokens[start+offset-prefix:start+offset+body]
            valid[row, dest:self.context+body] = True
            y[row, :body] = labels
        return torch.from_numpy(x), torch.from_numpy(valid), torch.from_numpy(y)

    def epoch_order(self, seed):
        """Each window with at least one target appears exactly once; no rejection sampling."""
        selected = []
        base = 0
        for tokens, index, cumulative, mask in self.shards:
            count = int(cumulative[-1])
            if mask is None:
                selected.append(np.arange(base, base+count, dtype=np.int64))
            else:
                local = np.arange(count, dtype=np.int64)
                docs = np.searchsorted(cumulative, local, side="right")
                previous = np.concatenate(([0], cumulative[:-1]))
                starts = index[docs].astype(np.int64)+(local-previous[docs])*self.block_size+1
                ends = np.minimum(starts+self.block_size, index[docs+1].astype(np.int64))
                prefix = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
                selected.append(base+local[prefix[ends] > prefix[starts]])
            base += count
        order = np.concatenate(selected)
        np.random.default_rng(seed).shuffle(order)
        return order

    def batch_windows(self, windows):
        width = self.context+self.block_size
        x = np.zeros((len(windows), width), dtype=np.int64)
        valid = np.zeros_like(x, dtype=np.bool_)
        y = np.full((len(windows), self.block_size), -100, dtype=np.int64)
        for row, window in enumerate(windows):
            sid = int(np.searchsorted(self.cumulative, window, side="right"))
            local = int(window)-(int(self.cumulative[sid-1]) if sid else 0)
            tokens, index, cumulative, mask = self.shards[sid]
            doc = int(np.searchsorted(cumulative, local, side="right"))
            chunk = local-(int(cumulative[doc-1]) if doc else 0)
            start, end = int(index[doc]), int(index[doc+1])
            offset = chunk*self.block_size
            body = min(self.block_size, end-start-1-offset)
            labels = np.array(tokens[start+offset+1:start+offset+body+1], dtype=np.int64)
            if mask is not None:
                labels[np.asarray(mask[start+offset+1:start+offset+body+1]) == 0] = -100
            prefix = min(self.context, offset)
            x[row, self.context-prefix:self.context+body] = tokens[start+offset-prefix:start+offset+body]
            valid[row, self.context-prefix:self.context+body] = True
            y[row, :body] = labels
        return torch.from_numpy(x), torch.from_numpy(valid), torch.from_numpy(y)
