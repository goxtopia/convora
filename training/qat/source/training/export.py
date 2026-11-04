"""Export trained CPU weights/table and a static one-token ONNX graph for later RKNN calibration."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.download import sha256
from training.model import ModelConfig, TCNLanguageModel
from training.train import load_checkpoint


class CachedStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # A frozen export snapshot: older ONNX exporters retain parameter-only
        # MatMul nodes even with constant folding enabled. Precompute E@P here
        # so the deployed graph has exactly the original single output Conv.
        self.register_buffer("head_weight", model.output_weight().detach().clone())

    @staticmethod
    def inject(h, memory, e):
        gate = F.hardsigmoid(F.conv2d(h, e.gwh.weight.unsqueeze(2)) +
                             F.conv2d(memory, e.gwm.weight.unsqueeze(2), e.gwm.bias))
        # Scale VALUE rows, after the nonlinear gate. Folds to a constant at export.
        value_weight = e.gv.weight * e.alpha.reshape(-1, 1, 1)
        return h + gate * F.conv2d(memory, value_weight.unsqueeze(2))

    def forward(self, x, *args):
        cfg = self.model.config
        h, hidden = x, []
        memory = args[-1] if self.model.engram else None
        count = len(self.model.blocks)
        for round_index in range(cfg.num_passes):
            for i, block in enumerate(self.model.blocks):
                taps = torch.cat((args[round_index*count+i], h), dim=3)
                temporal = F.conv2d(taps, block.weight.unsqueeze(2), block.bias)
                if cfg.block_style == "post_relu":
                    h = F.relu(h + temporal)
                else:
                    h = h + F.conv2d(F.relu(temporal), self.model.pointwise[i].weight.unsqueeze(2))
                if self.model.engram is not None and i == cfg.hook:
                    h = self.inject(h, memory, self.model.engram)
                if str(i) in self.model.extra_engram:
                    h = self.inject(h, memory, self.model.extra_engram[str(i)])
                if round_index*count+i < cfg.num_passes*count-1:
                    hidden.append(h)
        logits = F.conv2d(self.model.final_norm(h), self.head_weight.unsqueeze(2))
        return (logits, *hidden)


def export_cached_onnx(model, path, *, allow_dynamic_norm=False):
    if model.config.head_norm != "none" and not allow_dynamic_norm:
        raise ValueError("Dynamic RMSNorm has not been validated for RV1103 INT8. "
                         "Use head_norm=none, or explicitly allow it for experimental ONNX checks.")
    c = model.config.channels
    count = len(model.config.dilations)
    passes = model.config.num_passes
    inputs = (torch.zeros(1, c, 1, 1),) + tuple(torch.zeros(1, c, 1, 2) for _ in range(count*passes))
    if passes == 1:
        names = ["x"] + [f"state{i}" for i in range(count)]
        outputs = ["logits"]+[f"h{i}" for i in range(count-1)]
    else:
        names = ["x"] + [f"state_r{r}_{i}" for r in range(passes) for i in range(count)]
        outputs = ["logits"] + [f"h_r{r}_{i}" for r in range(passes) for i in range(count)
                                if r*count+i < passes*count-1]
    if model.engram:
        inputs += (torch.zeros(1, c, 1, 1),)
        names.append("memory")
    kwargs = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    # Deliberately retain the static opset-12 exporter used by RKNN 1.6, across PyTorch versions.
    torch.onnx.export(CachedStep(model), inputs, str(path), opset_version=12,
                      input_names=names, output_names=outputs,
                      do_constant_folding=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-dynamic-norm", action="store_true",
                        help="Experimental ONNX only; does not establish RV1103 quantization support")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = TCNLanguageModel(config).eval()
    model.load_state_dict(checkpoint["model"])
    if config.head_norm != "none" and not args.allow_dynamic_norm:
        parser.error("Dynamic RMSNorm is experimental; RV1103 exports use head_norm=none. "
                     "Use --allow-dynamic-norm only for an explicit experiment.")
    args.out.mkdir(parents=True, exist_ok=True)
    if model.engram:
        table = torch.cat([e.weight.detach() for e in model.engram.tables]).numpy()
        scales = np.maximum(np.abs(table).max(axis=1)/127., 1e-8).astype("<f4")
        quantized = np.clip(np.rint(table/scales[:, None]), -127, 127).astype("i1")
        quantized.tofile(args.out/"memory.i8")
        scales.tofile(args.out/"memory_scale.f32")
    # This file is the CPU INPUT lookup, not an extra trainable embedding. Bake
    # the tied-model input scale so readers must not multiply it a second time.
    (model.embedding.weight.detach() * model.embedding_scale).numpy().astype("<f4").tofile(args.out/"embedding.f32")
    # Keep the framework-independent original floating weights for calibration/reference generation.
    np.savez(args.out/"weights.npz", **{k: v.detach().numpy() for k, v in model.state_dict().items()})
    export_cached_onnx(model, args.out/"tcn_step.onnx", allow_dynamic_norm=args.allow_dynamic_norm)
    metadata = {"model": config.to_dict(), "parameters": model.parameter_counts(),
                "embedding_scale": model.embedding_scale,
                "embedding_input_scale_baked": True,
                "head_projection_folded": config.tied_head_projection,
                "dynamic_norm_experimental": config.head_norm != "none",
                "checkpoint_sha256": sha256(args.checkpoint), "tokenizer_sha256": checkpoint["tokenizer_sha256"],
                "trained_supervised_tokens": checkpoint["tokens_seen"],
                "memory_quantization": "per-row symmetric int8 + FP32 scale",
                "cache_layout": {"independent_passes": config.num_passes,
                                 "layers_per_pass": len(config.dilations),
                                 "int8_history_bytes": config.context*config.channels,
                                 "commits_per_cache_per_token": 1},
                "status": "Static ONNX exported; trained-weight RKNN calibration and board validation still required"}
    for key in ("chat_template", "generation_template"):
        if key in checkpoint:
            metadata[key] = checkpoint[key]
    (args.out/"config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
