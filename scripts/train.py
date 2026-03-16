from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from canos.data import (
    GraphBatch,
    SyntheticPowerDataset,
    graphbatch_from_pyg,
    patch_torch_geometric_tar_extract_compat,
)
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


def _build_dataset_and_loader(cfg: dict) -> tuple[Any, Any, GraphBatch]:
    dcfg = cfg["dataset"]
    dataset_type = dcfg.get("type", "synthetic")

    if dataset_type == "pyg_opf":
        patch_torch_geometric_tar_extract_compat()
        from torch_geometric.datasets import OPFDataset
        from torch_geometric.loader import DataLoader as PyGDataLoader

        dataset = OPFDataset(
            root=dcfg["root"],
            split=dcfg.get("split", "train"),
            case_name=dcfg.get("case_name", "pglib_opf_case500_goc"),
            num_groups=dcfg.get("num_groups", 1),
            topological_perturbations=dcfg.get("topological_perturbations", False),
            force_reload=dcfg.get("force_reload", False),
        )
        loader = PyGDataLoader(
            dataset,
            batch_size=dcfg.get("batch_size", 1),
            shuffle=dcfg.get("shuffle", True),
        )
        sample = graphbatch_from_pyg(dataset[0])
        return dataset, loader, sample

    dataset = SyntheticPowerDataset(
        size=dcfg["size"],
        n_bus=dcfg["n_bus"],
        n_gen=dcfg["n_gen"],
        n_load=dcfg["n_load"],
        n_shunt=dcfg["n_shunt"],
        n_line=dcfg["n_line"],
        n_trafo=dcfg["n_trafo"],
        seed=dcfg.get("seed", 0),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=dcfg.get("shuffle", True),
        collate_fn=lambda items: items[0],
    )
    sample = dataset[0]
    return dataset, loader, sample


def _to_graph_batch(batch: Any, device: torch.device) -> GraphBatch:
    if isinstance(batch, GraphBatch):
        return batch.to(device)
    if hasattr(batch, "node_types") and hasattr(batch, "edge_types"):
        return graphbatch_from_pyg(batch.to(device))
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CANOS reproduction")
    parser.add_argument("--config", type=str, default="configs/canos_small.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = torch.device(args.device)

    dataset, loader, sample = _build_dataset_and_loader(cfg)
    del dataset

    model = build_model(cfg, sample).to(device)
    tcfg = cfg["train"]
    optimizer = Adam(model.parameters(), lr=tcfg["lr"])

    model.train()
    steps = tcfg["steps"]
    constraint_weight = tcfg.get("constraint_weight", 0.1)

    start = time.time()
    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = _to_graph_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        losses = compute_total_loss(batch, pred, constraint_weight=constraint_weight)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
        optimizer.step()

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
