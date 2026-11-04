"""Single-GPU, interruptible one/two-pass TCN + Engram language-model training."""
import argparse
import hashlib
from contextlib import nullcontext
import json
import math
import os
import pickle
from pathlib import Path
import random
import signal
import time
import types
import warnings
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from training.data import TokenCorpus
from training.model import ModelConfig, TCNLanguageModel
from delta_model import DeltaModel
from prefix_data import FullPrefixCorpus
from micro_graph import MicroGraph


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path, device="cpu"):
    # Checkpoints contain Python RNG/optimizer state. Load only your own trusted checkpoints.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch 1.13 CPU smoke environment
        return torch.load(path, map_location=device)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing != "numpy._core" and not missing.startswith("numpy._core."):
            raise
        # NumPy 2 pickles RNG arrays via numpy._core; the RKNN 1.6 host needs NumPy 1.
        # Remap only this namespace in the trusted checkpoint unpickler, not global modules.
        class CompatibleUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "numpy._core" or module.startswith("numpy._core."):
                    module = "numpy.core" + module[len("numpy._core"):]
                return super().find_class(module, name)
        compatibility = types.ModuleType("numpy_checkpoint_compatibility")
        compatibility.Unpickler = CompatibleUnpickler
        compatibility.load = lambda stream, **kwargs: CompatibleUnpickler(stream, **kwargs).load()
        # PyTorch 1.13 overwrites its pickle_module argument; scope the module override to this read.
        with mock.patch.object(torch.serialization, "pickle", compatibility):
            return torch.load(path, map_location=device, pickle_module=compatibility, weights_only=False)


def training_cuda_device(device=None):
    """Resolve only the GPU used by this single-GPU trainer, never all visible GPUs."""
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())


def rng_state(device=None):
    gpu = training_cuda_device(device)
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": {"state": torch.cuda.get_rng_state(gpu).cpu(),
                     "logical_index": gpu.index,
                     "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
            if gpu is not None else None}


def restore_rng(state, device=None, legacy_cuda_rng_index=0):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    gpu = training_cuda_device(device)
    saved = state.get("cuda")
    if gpu is not None and saved is not None:
        if isinstance(saved, dict):
            # A training RNG stream follows the model, even when CUDA_VISIBLE_DEVICES remaps it.
            cuda_rng = saved["state"]
        else:
            # Version-1 checkpoints saved every visible GPU. Their default training device was
            # logical cuda:0; an explicitly selected old device can be specified by this option.
            if not 0 <= legacy_cuda_rng_index < len(saved):
                raise ValueError(f"Legacy CUDA RNG index {legacy_cuda_rng_index} is outside the "
                                 f"{len(saved)} saved states")
            cuda_rng = saved[legacy_cuda_rng_index]
            if len(saved) > 1 or legacy_cuda_rng_index != 0:
                warnings.warn(f"Legacy checkpoint: restoring saved logical cuda:{legacy_cuda_rng_index} "
                              f"RNG to current {gpu}; other saved GPU states are unused. If the old run "
                              "used --device cuda:N, pass --legacy-cuda-rng-index N.", stacklevel=2)
        torch.cuda.set_rng_state(cuda_rng.cpu(), device=gpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, default=Path("training-data/pretrain/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("training-runs/pilot"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--micro-batch", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--token-budget", type=int, default=5_000_000,
                        help="Actual supervised tokens, including repeats; stop at an update boundary")
    parser.add_argument("--epochs", type=int, default=0,
                        help="When positive, traverse all supervised windows once per epoch instead of random sampling")
    parser.add_argument("--save-epoch-checkpoints", action="store_true",
                        help="Retain epoch-NNNN.pt after each completed epoch; does not change the training contract")
    parser.add_argument("--teacher", type=Path, help="Verified same-tokenizer native pretrain teacher directory")
    parser.add_argument("--kd-weight", type=float, default=.5, help="Full-logits KL weight; remaining weight is data CE")
    parser.add_argument("--kd-temperature", type=float, default=2.)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=.1)
    parser.add_argument("--warmup-ratio", type=float, default=.03)
    parser.add_argument("--weight-decay", type=float, default=.1)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=110305)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save-seconds", type=float, default=120.)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", default="auto", help="auto, none, or trusted checkpoint path")
    parser.add_argument("--legacy-cuda-rng-index", type=int, default=0,
                        help="Training GPU's OLD logical index for version-1 multi-GPU RNG checkpoints")
    parser.add_argument("--init-from", type=Path, help="Weights only, e.g. pretraining -> SFT; resets optimizer/cursor")
    parser.add_argument("--no-engram", action="store_true", help="Same TCN without memory, for controlled comparison")
    parser.add_argument("--model-config", type=Path, help="Architecture-only ModelConfig JSON; omitted uses the legacy architecture")
    parser.add_argument("--passes", type=int, choices=[1, 2], default=None,
                        help="Weight-shared TCN passes; default 1 preserves old checkpoints")
    parser.add_argument("--stop-after-updates", type=int, default=0,
                        help="Graceful early exit for preemption/testing, without changing LR schedule")
    parser.add_argument("--deterministic", action="store_true")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        values = json.loads(preliminary.config.read_text(encoding="utf-8"))
        valid_keys = {a.dest for a in parser._actions}
        unknown = set(values)-valid_keys
        if unknown:
            parser.error(f"Unknown config keys: {unknown}")
        parser.set_defaults(**values)
    args = parser.parse_args()
    args.data, args.out = Path(args.data), Path(args.out)
    if args.init_from:
        args.init_from = Path(args.init_from)
    if args.teacher:
        args.teacher = Path(args.teacher)
    if min(args.micro_batch, args.accumulation, args.block_size, args.token_budget,
           args.threads, args.eval_batches, args.eval_every, args.log_every) < 1:
        parser.error("Batch sizes, token budget, threads, evaluation and logging intervals must be positive")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1 or args.lr <= 0:
        parser.error("Invalid learning-rate schedule")
    if args.epochs < 0:
        parser.error("epochs must be nonnegative")
    if not 0 <= args.kd_weight <= 1 or args.kd_temperature <= 0:
        parser.error("Invalid distillation weight/temperature")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu --precision fp32 for smoke tests")
    if device.type == "cuda":
        device = training_cuda_device(device)
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(device).total_memory, device)
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --precision fp16 (e.g. T4)")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = not args.deterministic
        torch.backends.cudnn.allow_tf32 = not args.deterministic
        torch.backends.cudnn.benchmark = not args.deterministic
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    config = ModelConfig.from_dict(json.loads(Path(args.model_config).read_text(encoding="utf-8"))) \
        if args.model_config else ModelConfig()
    if args.no_engram:
        config.use_engram = False
    if args.passes is not None:
        config.num_passes = args.passes
    config.__post_init__()
    args.passes = config.num_passes
    assert args.precision == "bf16" and args.epochs == 1 and args.teacher is None and args.init_from is None
    assert args.micro_batch == 128 and args.accumulation == 2 and args.block_size == 256
    model = DeltaModel(config).to(device)
    model.delta.backend = 'fused'
    def state_sha(values):
        digest = hashlib.sha256()
        for name, value in sorted(values.items()):
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    initial_backbone_sha = state_sha({k:v for k,v in model.state_dict().items() if not k.startswith('delta.')})
    assert initial_backbone_sha == '6f92802fc867f2da027833a7be355dd20eb04d208a7fea12a7f942a8e7801c37'
    assert state_sha(model.state_dict()) == '29390dcd2747fcd10c443cb25b3c7c5420897d967c70feb0287f048fa8776373'
    assert sum(p.numel() for p in model.parameters()) == 9289608
    train = FullPrefixCorpus(args.data, "train", args.block_size, config.context)
    assert train.manifest["fingerprint"] == "316a97a0f816eeacbf7a9ca79646e90dd971c2be842d849bdea1f8c9970ab078"
    val = TokenCorpus(args.data, "val", args.block_size, config.context)
    if train.manifest["vocab_size"] != config.vocab_size:
        raise ValueError("Dataset vocabulary does not match model")
    teacher = teacher_identity = None
    if args.teacher:
        if train.manifest["mode"] != "text":
            raise ValueError("This KD stage is for plain pretraining text")
        from training.logits_kd import load_teacher
        teacher, teacher_identity = load_teacher(args.teacher, train.manifest["tokenizer_sha256"], device)
    # named_parameters deduplicates the tied input/output Parameter. Channel alpha
    # has broadcast shape (1,C,1), but is a gain vector rather than a weight matrix.
    named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named_parameters]
    def decay(name, p):
        return p.ndim >= 2 and not name.endswith(".alpha")
    groups = [{"params": [p for name, p in named_parameters if decay(name, p)], "weight_decay": args.weight_decay},
              {"params": [p for name, p in named_parameters if not decay(name, p)], "weight_decay": 0.}]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(.9, .95), eps=1e-8)
    scaled = device.type == "cuda" and args.precision == "fp16"
    scaler = (torch.amp.GradScaler("cuda", enabled=scaled)
              if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=scaled))
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    def autocast():
        return torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" and args.precision != "fp32" else nullcontext()
    args.out.mkdir(parents=True, exist_ok=True)
    contract_keys = ("micro_batch", "accumulation", "block_size", "token_budget", "lr", "min_lr_ratio",
                     "warmup_ratio", "weight_decay", "grad_clip", "seed", "precision", "deterministic", "passes")
    contract = {key: getattr(args, key) for key in contract_keys}
    contract["epochs"] = args.epochs
    contract.update(architecture='tcn-engram-fp-delta-lite-4x32', delta_backend='fused', qat=False,
                    full_prefix_width=768, compiled_loss=True, validation_micro_batch=64,
                    source_manifest_sha256=hashlib.sha256((Path(__file__).resolve().parents[2]/'source-manifest.json').read_bytes()).hexdigest())
    if teacher is not None:
        contract.update(teacher=teacher_identity, kd_weight=args.kd_weight, kd_temperature=args.kd_temperature)
    step = tokens_seen = micro_cursor = 0
    epoch_cursor = 0
    order = train.epoch_order(args.seed) if args.epochs else None
    epoch_windows = len(order) if order is not None else 0
    if args.epochs and not epoch_windows:
        raise ValueError("No supervised windows in epoch")
    total_windows = epoch_windows*args.epochs
    assert epoch_windows == 2411917
    assert hashlib.sha256(order.tobytes()).hexdigest() == 'f3b6d81f19038fdf9b90b250080865fc66b237d7b0bc6c164dcc0a562ed3b64c'
    initial_metadata = None
    best_val = math.inf
    checkpoint = None
    resume_path = args.out/"last.pt" if args.resume == "auto" else (None if args.resume == "none" else Path(args.resume))
    if args.resume == "none" and (args.out/"last.pt").exists():
        raise ValueError("Output already contains last.pt; choose a fresh --out or resume it")
    if args.init_from and resume_path and resume_path.exists():
        raise ValueError("Choose resume OR init-from, not both; use --resume none for a new stage")
    if resume_path and resume_path.exists():
        checkpoint = load_checkpoint(resume_path)
        saved_contract = dict(checkpoint["contract"])
        saved_contract.setdefault("passes", 1)  # backward-compatible single-pass checkpoints
        saved_contract.setdefault("epochs", 0)
        if saved_contract != contract or checkpoint["data_fingerprint"] != train.manifest["fingerprint"]:
            raise ValueError("Resume data/training settings changed; use the original settings or a new weights-only stage")
        if ModelConfig.from_dict(checkpoint["model_config"]) != config:
            raise ValueError("Resume model configuration changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step, tokens_seen, micro_cursor = checkpoint["step"], checkpoint["tokens_seen"], checkpoint["micro_cursor"]
        best_val = checkpoint["best_val"]
        epoch_cursor = checkpoint.get("epoch_cursor", 0)
        initial_metadata = checkpoint.get("initialization")
        if args.epochs:
            order = train.epoch_order(args.seed+min(epoch_cursor//epoch_windows, args.epochs-1))
        restore_rng(checkpoint["rng"], device, args.legacy_cuda_rng_index)
        print(f"Resumed update={step}, tokens={tokens_seen:,}, micro_cursor={micro_cursor}", flush=True)
    elif args.resume not in ("auto", "none"):
        raise FileNotFoundError(resume_path)
    elif args.init_from:
        initial = load_checkpoint(args.init_from)
        source_config = ModelConfig.from_dict(initial["model_config"])
        source_weights_config, target_weights_config = source_config.to_dict(), config.to_dict()
        source_weights_config.pop("num_passes")
        target_weights_config.pop("num_passes")
        if source_weights_config != target_weights_config:
            raise ValueError("Initial model weight configuration differs")
        if initial["tokenizer_sha256"] != train.manifest["tokenizer_sha256"]:
            raise ValueError("Initial checkpoint used a different tokenizer")
        model.load_state_dict(initial["model"])
        from training.download import sha256
        initial_metadata = {"path": str(args.init_from), "sha256": sha256(args.init_from),
                            "tokens_seen": initial.get("tokens_seen"), "step": initial.get("step")}
        if source_config.num_passes != config.num_passes:
            print(f"Weights-only warm start: {source_config.num_passes} -> {config.num_passes} passes; "
                  "optimizer and token progress reset", flush=True)
    elif (args.out/"last.pt").exists():
        raise ValueError("Output already contains a checkpoint; choose a fresh --out or resume it")

    interrupted = {"signal": None}
    def request_stop(signum, _frame):
        interrupted["signal"] = signum
        print(f"Signal {signum}: finishing current optimizer update, then saving", flush=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def checkpoint_payload():
        return {"format_version": 3, "architecture": "tcn-engram-fp-delta-lite-4x32",
                "delta_config": {"heads":4,"dimension":32,"normalize":"bounded","qat":False},
                "training_elapsed_seconds": prior_elapsed+time.monotonic()-started,
                "model_config": config.to_dict(), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(device),
                "contract": contract, "data_fingerprint": train.manifest["fingerprint"],
                "tokenizer_sha256": train.manifest["tokenizer_sha256"], "step": step,
                "tokens_seen": tokens_seen, "micro_cursor": micro_cursor, "best_val": best_val,
                "epoch_cursor": epoch_cursor, "epoch_windows": epoch_windows,
                "initialization": initial_metadata, "chat_template": train.manifest.get("chat_template"),
                "torch_version": torch.__version__}

    def save(name="last.pt"):
        atomic_save(checkpoint_payload(), args.out/name)

    @torch.no_grad()
    def evaluate():
        model.eval()
        sums = {"normal": 0., "zero_memory": 0.}
        count = 0
        for i in range(args.eval_batches):
            x, valid, y = val.batch(64, args.seed+10**12+i)
            x, valid, y = (t.to(device) for t in (x, valid, y))
            count += int((y != -100).sum())
            for mode in (sums if model.engram else ("normal",)):
                with autocast():
                    logits = model(x, valid, config.context, memory_enabled=mode == "normal")
                    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1),
                                           ignore_index=-100, reduction="sum")
                sums[mode] += float(loss)
        model.train()
        assert count == 230424 if args.eval_batches == 20 else count > 0
        return {"loss": sums["normal"]/count, "ppl": math.exp(min(sums["normal"]/count, 50)),
                "zero_memory_loss": sums["zero_memory"]/count if model.engram else None,
                "supervised_tokens": count}

    metadata = {"model": config.to_dict(), "parameters": model.parameter_counts(),
                "contract": contract, "data_fingerprint": train.manifest["fingerprint"],
                "device": str(device), "torch_version": torch.__version__,
                "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    metadata.update(epoch_windows=epoch_windows, total_windows=total_windows, initialization=initial_metadata)
    (args.out/"run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    rng_before_graph = rng_state(device)
    graph_started = time.monotonic()
    engine = MicroGraph(model, args.micro_batch, compiled_loss=True)
    optimizer.zero_grad(set_to_none=False)
    restore_rng(rng_before_graph, device)
    prior_elapsed = checkpoint.get('training_elapsed_seconds',0.) if checkpoint else 0.
    print(json.dumps({'event':'graph_ready','seconds':time.monotonic()-graph_started,
                      'resumed_step':step,'initial_backbone_sha256':initial_backbone_sha}),flush=True)
    last_save = started = last_log = time.monotonic()
    log_tokens, log_loss, log_count = tokens_seen, 0., 0
    log_ce = log_kl = log_teacher_ce = 0.
    metrics_file = open(args.out/"metrics.jsonl", "a", encoding="utf-8", buffering=1)
    def emit(record):
        record = dict(record, step=step, tokens=tokens_seen)
        metrics_file.write(json.dumps(record) + "\n")
        status = dict(status='running' if record.get('event') not in ('finished','stopped') else record['event'],
                      updated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                      epoch_cursor=epoch_cursor,epoch_windows=epoch_windows,
                      training_elapsed_seconds=prior_elapsed+time.monotonic()-started,
                      graph_replays=engine.replays,
                      allocated_peak_gib=torch.cuda.max_memory_allocated()/1024**3,
                      reserved_peak_gib=torch.cuda.max_memory_reserved()/1024**3,**record)
        temporary=args.out/'status.json.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(args.out/'status.json')
        print(json.dumps(record), flush=True)
    model.train()
    overflow_retries = 0
    try:
        while (epoch_cursor < total_windows if args.epochs else tokens_seen < args.token_budget) and not interrupted["signal"]:
            # Prefetch only the current update; labels determine exact token-weighted accumulation.
            if args.epochs:
                epoch = epoch_cursor//epoch_windows
                offset = epoch_cursor % epoch_windows
                if offset == 0:
                    order = train.epoch_order(args.seed+epoch)
                stop = min(epoch_windows, offset+args.micro_batch*args.accumulation)
                batches = [train.batch_windows(order[i:min(i+args.micro_batch, stop)])
                           for i in range(offset, stop, args.micro_batch)]
                update_windows = stop-offset
                schedule_budget, schedule_seen = total_windows, epoch_cursor
                schedule_update = update_windows
            else:
                batches = [train.batch(args.micro_batch, args.seed+micro_cursor+i) for i in range(args.accumulation)]
                schedule_budget, schedule_seen = args.token_budget, tokens_seen
                schedule_update = None
            update_tokens = sum(int((y != -100).sum()) for _, _, y in batches)
            if not update_tokens:
                raise ValueError("Empty optimizer update")
            warmup = max(1, schedule_budget*args.warmup_ratio)
            progress = min(1., max(0., (schedule_seen-warmup)/max(1, schedule_budget-warmup)))
            factor = min(1., (schedule_seen+(schedule_update if args.epochs else update_tokens))/warmup) if schedule_seen < warmup else (
                args.min_lr_ratio+(1-args.min_lr_ratio)*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr*factor
            optimizer.zero_grad(set_to_none=False)
            loss_sum = 0.
            ce_sum = kl_sum = teacher_ce_sum = 0.
            for x, valid, y in batches:
                x, valid, y = (t.to(device) for t in (x, valid, y))
                if teacher is None:
                    loss_sum += engine.backward(x, valid, y, update_tokens)
                    continue
                with autocast():
                    logits = model(x, valid, config.context)
                    if teacher is None:
                        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1),
                                               ignore_index=-100, reduction="sum")
                    else:
                        from training.logits_kd import target_teacher_logits, full_logit_loss
                        teacher_targets = target_teacher_logits(teacher, x, valid, y, config.context)
                        loss, ce, kl, teacher_ce = full_logit_loss(logits, teacher_targets, y,
                            args.kd_weight, args.kd_temperature)
                        ce_sum += float(ce.detach())
                        kl_sum += float(kl.detach())
                        teacher_ce_sum += float(teacher_ce.detach())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite loss; previous checkpoint retained")
                scaler.scale(loss/update_tokens).backward()
                loss_sum += float(loss.detach())
            scaler.unscale_(optimizer)
            try:
                norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
            except RuntimeError as exc:
                nonfinite = any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters)
                if scaler.is_enabled() and nonfinite and overflow_retries < 16:
                    # FP16 dynamic scaling can overflow initially. Retry the SAME data cursor;
                    # GradScaler skips this optimizer step and reduces the scale.
                    scaler.step(optimizer)
                    scaler.update()
                    overflow_retries += 1
                    emit({"event": "fp16_overflow_retry", "scale": scaler.get_scale(),
                          "attempt": overflow_retries})
                    continue
                raise FloatingPointError("Invalid gradient norm; previous checkpoint retained") from exc
            scaler.step(optimizer)
            scaler.update()
            overflow_retries = 0
            step += 1
            micro_cursor += args.accumulation
            if args.epochs:
                epoch_cursor += update_windows
            tokens_seen += update_tokens
            log_loss += loss_sum
            log_count += update_tokens
            log_ce += ce_sum
            log_kl += kl_sum
            log_teacher_ce += teacher_ce_sum
            now = time.monotonic()
            if step % args.log_every == 0 or step == 1:
                emit({"event": "train", "loss": log_loss/log_count, "lr": args.lr*factor,
                      "grad_norm": float(norm), "tokens_per_second": (tokens_seen-log_tokens)/max(now-last_log, .001),
                      "elapsed_seconds": now-started,
                      "epochs_completed": epoch_cursor/epoch_windows if args.epochs else None,
                      "cuda_peak_gib": torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else None})
                if teacher is not None:
                    emit(dict(event="distillation",data_ce=log_ce/log_count,full_vocab_kl=log_kl/log_count,
                              teacher_ce=log_teacher_ce/log_count,kd_weight=args.kd_weight,
                              kd_temperature=args.kd_temperature))
                    log_ce = log_kl = log_teacher_ce = 0.
                last_log, log_tokens, log_loss, log_count = now, tokens_seen, 0., 0
            if now-last_save >= args.save_seconds:
                save()
                last_save = time.monotonic()
            if step % args.eval_every == 0 and not interrupted["signal"]:
                result = evaluate()
                emit(dict(result, event="validation"))
                if result["loss"] < best_val:
                    best_val = result["loss"]
                    save("best.pt")
                save()
                last_save = time.monotonic()
            if args.save_epoch_checkpoints and args.epochs and epoch_cursor % epoch_windows == 0:
                name = f"epoch-{epoch_cursor//epoch_windows:04d}.pt"
                save(name)
                emit({"event": "epoch_snapshot", "checkpoint": name,
                      "epochs_completed": epoch_cursor/epoch_windows})
            if args.stop_after_updates and step >= args.stop_after_updates:
                break
        save()
        # A preemption saves immediately, without waiting for an expensive final validation.
        if not interrupted["signal"]:
            result = evaluate()
            emit(dict(result, event="validation_final"))
            if result["loss"] < best_val:
                best_val = result["loss"]
                save("best.pt")
            save()
        emit({"event": "stopped" if interrupted["signal"] else "finished",
              "budget_reached": epoch_cursor >= total_windows if args.epochs else tokens_seen >= args.token_budget,
              "epochs_completed": epoch_cursor/epoch_windows if args.epochs else None,
              "signal": interrupted["signal"]})
    finally:
        metrics_file.close()


if __name__ == "__main__":
    main()
