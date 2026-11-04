"""Reproducible forensics for the Convora deep-analysis report (ANALYSIS_REPORT.md).

Recomputes the headline numbers cited in the report:

  1. TiedHead P orthogonalization statistics (self-echo suppression, Sec. 3.1)
  2. TCN per-layer 3-tap filter energy distribution (history-tap collapse, Sec. 3.3)
  3. Engram gate energy allocation + QAT zero-gradient bucket archaeology (Sec. 3.4)
  4. IEEE-754 float32 -127.5 zero-point rounding boundary (V-projection 1-LSB bias, Sec. 5.2)

Usage:
    python3 reports/mechanistic_analysis/reproduce_analysis.py

Requires the released artifacts under models/ (see scripts/verify_release.py).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modeling"))

from load import load_float


def effective_rank(singular_values: np.ndarray) -> float:
    p = singular_values / (singular_values.sum() + 1e-12)
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def main():
    b_sft, _, _ = load_float(ROOT / "models/float/sft-before-qat.pt")
    b_qat, _, saved_qat = load_float(ROOT / "models/float/qat-final.pt")

    # 1. TiedHead P orthogonalization analysis
    E = b_qat.embedding.weight.detach().numpy()
    P = b_qat.head.proj.weight.detach().squeeze(-1).numpy()
    EP = E @ P
    cos_e_ep = np.sum(E * EP, axis=1) / (np.linalg.norm(E, axis=1) * np.linalg.norm(EP, axis=1) + 1e-12)
    _, s_p, _ = np.linalg.svd(P)
    _, s_e, _ = np.linalg.svd(E, full_matrices=False)

    # 2. TCN tap energy distribution
    tcn_tap_energy = []
    for i, (block, d) in enumerate(zip(b_qat.blocks, b_qat.config.dilations)):
        w = block.weight.detach().numpy()
        e = [float(np.sum(w[:, :, k] ** 2)) for k in range(3)]
        tot = sum(e)
        tcn_tap_energy.append(
            {
                "layer": i,
                "dilation": int(d),
                "tap_t_minus_2d_pct": round(e[0] / tot * 100, 2),
                "tap_t_minus_d_pct": round(e[1] / tot * 100, 2),
                "tap_t_pct": round(e[2] / tot * 100, 2),
            }
        )

    # 3. Engram projection energy & QAT weight-decay forensics
    gwm_w = b_qat.engram.gwm.weight.detach().squeeze(-1).numpy()
    gv_w = b_qat.engram.gv.weight.detach().squeeze(-1).numpy()
    engram_tables_summary = []
    # QAT ran 5,902 steps at lr=1e-5 with wd=0.1 -> per-step multiplicative decay 1e-6.
    expected_qat_decay = (1.0 - 1e-5 * 0.1) ** 5902
    for ti in range(4):
        c0, c1 = ti * 64, (ti + 1) * 64
        n_sft = np.linalg.norm(b_sft.engram.tables[ti].weight.detach().numpy(), axis=1)
        n_qat = np.linalg.norm(b_qat.engram.tables[ti].weight.detach().numpy(), axis=1)
        ratio = n_qat / (n_sft + 1e-12)
        untouched = int(np.isclose(ratio, expected_qat_decay, atol=1e-4).sum())
        engram_tables_summary.append(
            {
                "table": ti,
                "order": int(b_qat.config.orders[ti]),
                "buckets": int(len(n_qat)),
                "gwm_energy_pct": round(float(np.sum(gwm_w[:, c0:c1] ** 2) / np.sum(gwm_w**2) * 100), 2),
                "gv_energy_pct": round(float(np.sum(gv_w[:, c0:c1] ** 2) / np.sum(gv_w**2) * 100), 2),
                "qat_unvisited_buckets": untouched,
                "qat_unvisited_pct": round(untouched / len(n_qat) * 100, 2),
            }
        )

    # 4. IEEE-754 float32 -127.5 rounding boundary check (export/compatible.py zero-points)
    ieee_boundary = {}
    for name, M in [("query_key", 1.0 / math.sqrt(32)), ("value", 4.0)]:
        low = torch.tensor([-M], dtype=torch.float32)
        high = torch.tensor([M], dtype=torch.float32)
        scale = ((high - low) / 255).clamp_min(1e-8)
        ratio = low / scale
        zp = (-128 - torch.round(ratio)).clamp(-128, 127)
        ieee_boundary[name] = {
            "scale_f32": float(scale.item()),
            "low_div_scale_f32": float(ratio.item()),
            "rounded_f32": float(torch.round(ratio).item()),
            "signed_zp": int(zp.item()),
            "uint8_zp": int(zp.item() + 128),
        }

    summary = {
        "model": saved_qat["model_config"],
        "tied_head_projection": {
            "P_effective_rank": round(effective_rank(s_p), 2),
            "E_effective_rank": round(effective_rank(s_e), 2),
            "cos_E_EP_mean": round(float(cos_e_ep.mean()), 6),
            "cos_E_EP_std": round(float(cos_e_ep.std()), 6),
            "self_logit_without_P_mean": round(float((np.sum(E * E, axis=1) * 16.0).mean()), 3),
            "self_logit_with_P_mean": round(float((np.sum(EP * E, axis=1) * 16.0).mean()), 3),
        },
        "tcn_tap_energy": tcn_tap_energy,
        "engram_tables_summary": engram_tables_summary,
        "ieee754_zp_boundary": ieee_boundary,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
