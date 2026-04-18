from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

SRC_DIR = (Path(__file__).resolve().parent.parent / "src").as_posix()
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from canos.data import GraphBatch
from canos.losses import compute_total_loss
from canos.model import CANOS


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _infer_feature_dims(sample: GraphBatch) -> dict:
    return {
        "bus_in": sample.bus_x.shape[-1],
        "gen_in": sample.gen_x.shape[-1],
        "load_in": sample.load_x.shape[-1],
        "shunt_in": sample.shunt_x.shape[-1],
        "line_in": sample.line_x.shape[-1],
        "trafo_in": sample.trafo_x.shape[-1],
    }


def build_model(cfg: dict, sample: GraphBatch) -> CANOS:
    mcfg = cfg["model"]
    dims = _infer_feature_dims(sample)
    return CANOS(
        bus_in=dims["bus_in"],
        gen_in=dims["gen_in"],
        load_in=dims["load_in"],
        shunt_in=dims["shunt_in"],
        line_in=dims["line_in"],
        trafo_in=dims["trafo_in"],
        hidden_size=mcfg["hidden_size"],
        num_message_passing_steps=mcfg["num_message_passing_steps"],
        decoder_hidden=mcfg["decoder_hidden"],
    )


def _build_paper_like_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    warmup_steps: int,
    decay_rate: float,
    transition_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step <= 0:
            return 0.0
        if step < warmup_steps:
            return step / float(warmup_steps)
        k = (step - warmup_steps) // int(transition_steps)
        return float(decay_rate**k)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _build_dataset_and_loader(cfg: dict) -> tuple[Any, Any, GraphBatch]:
    dcfg = cfg["dataset"]
    if dcfg.get("type", "sharaf_csv") != "sharaf_csv":
        raise ValueError("Only dataset.type: sharaf_csv is supported in this repo.")

    from canos.sharaf_csv import SharafCSVDataset

    dataset = SharafCSVDataset(
        root=dcfg["root"],
        sbase_mva=float(dcfg.get("sbase_mva", 100.0)),
        repeat=int(dcfg.get("repeat", 1)),
        sample_dirs=dcfg.get("sample_dirs"),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=dcfg.get("shuffle", True),
        collate_fn=lambda items: items[0],
    )
    sample = dataset[0]
    return dataset, loader, sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CANOS on Sharaf CSV exports")
    parser.add_argument("--config", type=str, default="configs/canos_sharaf_case57.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=None, help="Override cfg['train']['steps'] for quick smoke tests")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = torch.device(args.device)

    _, loader, sample = _build_dataset_and_loader(cfg)

    model = build_model(cfg, sample).to(device)
    tcfg = cfg["train"]
    optimizer = Adam(model.parameters(), lr=tcfg["lr"])
    scheduler = None
    if tcfg.get("lr_warmup_steps") is not None:
        scheduler = _build_paper_like_lr_scheduler(
            optimizer,
            base_lr=tcfg["lr"],
            warmup_steps=int(tcfg["lr_warmup_steps"]),
            decay_rate=float(tcfg.get("lr_decay_rate", 0.9)),
            transition_steps=int(tcfg.get("lr_decay_transition_steps", 4000)),
        )

    model.train()
    steps = args.steps if args.steps is not None else tcfg["steps"]
    constraint_weight = tcfg.get("constraint_weight", 0.1)

    start = time.time()
    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        losses = compute_total_loss(batch, pred, constraint_weight=constraint_weight)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % tcfg.get("log_every", 20) == 0 or step == 1:
            elapsed = time.time() - start
            print(
                f"step={step:05d} total={losses['total'].item():.6f} "
                f"sup={losses['supervised'].item():.6f} "
                f"cons={losses['constraints_total'].item():.6f} "
                f"time={elapsed:.1f}s"
            )

    out_dir = Path(tcfg.get("out_dir", "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "canos_repro.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
