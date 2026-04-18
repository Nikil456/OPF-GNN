#!/usr/bin/env python3
"""
Smoke test: CANOS forward + compute_total_loss on one Sharaf CSV sample.

Run from repo root (requires data_from_sharaf/ with at least one sample folder):
  python scripts/verify_canos_outputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import torch

from canos.losses import compute_total_loss
from canos.model import CANOS
from canos.sharaf_csv import SharafCSVDataset


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "data_from_sharaf"
    if not root.is_dir():
        raise SystemExit(f"Missing {root}: add CSV samples (see data_from_sharaf/README.md)")

    ds = SharafCSVDataset(root=str(root), sbase_mva=100.0, repeat=1)
    g = ds[0]
    nb, ng, ne, nt = g.bus_x.shape[0], g.gen_x.shape[0], g.line_x.shape[0], g.trafo_x.shape[0]

    model = CANOS(
        bus_in=g.bus_x.shape[-1],
        gen_in=g.gen_x.shape[-1],
        load_in=g.load_x.shape[-1],
        shunt_in=g.shunt_x.shape[-1],
        line_in=g.line_x.shape[-1],
        trafo_in=g.trafo_x.shape[-1],
        hidden_size=32,
        num_message_passing_steps=3,
        decoder_hidden=64,
    )
    pred = model(g)

    expected = {
        "bus_va": (nb,),
        "bus_vm": (nb,),
        "gen_pg": (ng,),
        "gen_qg": (ng,),
        "line_pf": (ne,),
        "line_qf": (ne,),
        "line_pt": (ne,),
        "line_qt": (ne,),
        "trafo_pf": (nt,),
        "trafo_qf": (nt,),
        "trafo_pt": (nt,),
        "trafo_qt": (nt,),
    }
    for key, shape in expected.items():
        assert key in pred, f"missing output key: {key}"
        assert pred[key].shape == shape, f"{key}: got {tuple(pred[key].shape)}, want {shape}"
        assert torch.isfinite(pred[key]).all(), f"non-finite values in {key}"

    losses = compute_total_loss(g, pred, constraint_weight=0.1)
    for name in ("total", "supervised", "constraints_total"):
        assert name in losses
        assert torch.isfinite(losses[name]).all(), f"non-finite {name}"

    losses["total"].backward()
    print("verify_canos_outputs: OK (forward, loss, backward)")


if __name__ == "__main__":
    main()
