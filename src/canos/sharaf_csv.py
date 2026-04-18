"""
Build GraphBatch from Sharaf-exported CSV folders (e.g. data_from_sharaf/.../sample_*/).

Units:
  - Powers in CSV (MW) are converted to per-unit using ``sbase_mva`` (default 100 MVA).
  - Angles in degrees are converted to radians for ``bus_va`` targets and edge limits.
  - Branch ``r_pu``, ``x_pu`` are already in pu on the system base.

Trafo edges: none in current Sharaf exports → empty ``trafo_*`` tensors (length 0).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .data import (
    BUS_TYPE_IDX,
    BUS_VMAX_IDX,
    BUS_VMIN_IDX,
    EDGE_ANGMAX_IDX,
    EDGE_ANGMIN_IDX,
    EDGE_BR_R_IDX,
    EDGE_BR_X_IDX,
    EDGE_BFR_IDX,
    EDGE_BTO_IDX,
    EDGE_RATE_A_IDX,
    GEN_PMAX_IDX,
    GEN_PMIN_IDX,
    GEN_QMAX_IDX,
    GEN_QMIN_IDX,
    GraphBatch,
    LOAD_PD_IDX,
    LOAD_QD_IDX,
    SHUNT_BS_IDX,
    SHUNT_GS_IDX,
    TRAFO_SHIFT_IDX,
    TRAFO_TAP_IDX,
)


def _read_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: str) -> float:
    return float(x) if x not in ("", None) else 0.0


def _bus_type_code(bus_type: str) -> float:
    t = (bus_type or "").lower().strip()
    if t == "slack":
        return 3.0
    if t == "generator":
        return 2.0
    if t == "load":
        return 1.0
    return 1.0


def _gen_dim() -> int:
    """Generator feature width (indices 0–6 used for limits)."""
    return 11


def _trafo_dim() -> int:
    """Transformer edge_attr width (tap/shift at 7,8); empty edges still define width."""
    return max(TRAFO_SHIFT_IDX, TRAFO_TAP_IDX) + 1


def graphbatch_from_sharaf_dir(sample_dir: Path, *, sbase_mva: float = 100.0) -> GraphBatch:
    """
    Load one sample folder containing ``buses.csv``, ``branches.csv``, ``generators.csv``.

    ``sample_dir`` may be a path like
    ``.../data_from_sharaf/pglib_opf_case57_ieee_sample_000009``.
    """
    sample_dir = Path(sample_dir)
    buses = _read_csv(sample_dir / "buses.csv")
    branches = _read_csv(sample_dir / "branches.csv")
    gens = _read_csv(sample_dir / "generators.csv")

    sbase = float(sbase_mva)
    if sbase <= 0:
        raise ValueError("sbase_mva must be positive")

    bus_id_to_idx: Dict[str, int] = {}
    for row in buses:
        bid = str(int(float(row["bus_id"])))
        bus_id_to_idx[bid] = int(row["node_index"])

    n_bus = len(buses)
    # bus_x: [pd_pu, type, vmin, vmax] (four columns per bus)
    pd_pu = torch.zeros(n_bus, dtype=torch.float32)
    bus_type_col = torch.zeros(n_bus, dtype=torch.float32)
    vmin = torch.full((n_bus,), 0.94, dtype=torch.float32)
    vmax = torch.full((n_bus,), 1.06, dtype=torch.float32)

    for row in buses:
        i = int(row["node_index"])
        pd_pu[i] = _f(row["Pd_mw"]) / sbase
        bus_type_col[i] = _bus_type_code(row["bus_type"])

    bus_x = torch.stack(
        [pd_pu, bus_type_col, vmin, vmax],
        dim=1,
    )

    # Loads: one node per bus with net load (PQ); Q not in CSV → 0
    load_rows: List[int] = []
    load_pd_list: List[float] = []
    for row in buses:
        i = int(row["node_index"])
        pd_mw = _f(row["Pd_mw"])
        if pd_mw > 1e-9 or int(float(row.get("is_load_bus", 0) or 0)) == 1:
            load_rows.append(i)
            load_pd_list.append(pd_mw / sbase)

    n_load = len(load_rows)
    load_x = torch.zeros((n_load, 2), dtype=torch.float32)
    for j, bus_i in enumerate(load_rows):
        load_x[j, LOAD_PD_IDX] = load_pd_list[j]
        load_x[j, LOAD_QD_IDX] = 0.0
    load_bus = torch.tensor(load_rows, dtype=torch.long)

    # Generators
    n_gen = len(gens)
    gen_x = torch.zeros((n_gen, _gen_dim()), dtype=torch.float32)
    gen_bus_list: List[int] = []
    for j, row in enumerate(gens):
        bid = str(int(float(row["bus_id"])))
        gen_bus_list.append(bus_id_to_idx[bid])
        gen_x[j, GEN_PMIN_IDX] = _f(row["Pmin_mw"]) / sbase
        gen_x[j, GEN_PMAX_IDX] = _f(row["Pmax_mw"]) / sbase
        gen_x[j, GEN_QMIN_IDX] = -2.0
        gen_x[j, GEN_QMAX_IDX] = 2.0
    gen_bus = torch.tensor(gen_bus_list, dtype=torch.long)

    # Shunts: none in Sharaf export
    n_shunt = 0
    shunt_x = torch.zeros((0, 2), dtype=torch.float32)
    shunt_bus = torch.zeros((0,), dtype=torch.long)

    # Branches (lines only); skip disabled
    line_from_list: List[int] = []
    line_to_list: List[int] = []
    line_attr_rows: List[List[float]] = []
    for row in branches:
        if int(float(row.get("line_enabled", 1) or 1)) != 1:
            continue
        fa = str(int(float(row["fbus"])))
        ta = str(int(float(row["tbus"])))
        fi, ti = bus_id_to_idx[fa], bus_id_to_idx[ta]
        angmin = math.radians(_f(row["angmin_deg"]))
        angmax = math.radians(_f(row["angmax_deg"]))
        rpu = _f(row["r_pu"])
        xpu = _f(row["x_pu"])
        b_ch = _f(row["b_susceptance"])
        b_half = 0.5 * b_ch
        rate_mva = _f(row["rateA_mva"])
        rate_pu = rate_mva / sbase
        line_attr_rows.append(
            [angmin, angmax, b_half, b_half, rpu, xpu, rate_pu],
        )
        line_from_list.append(fi)
        line_to_list.append(ti)

    n_line = len(line_from_list)
    line_x = torch.tensor(line_attr_rows, dtype=torch.float32)
    line_from = torch.tensor(line_from_list, dtype=torch.long)
    line_to = torch.tensor(line_to_list, dtype=torch.long)

    # No transformers in export
    td = _trafo_dim()
    trafo_x = torch.zeros((0, td), dtype=torch.float32)
    trafo_from = torch.zeros((0,), dtype=torch.long)
    trafo_to = torch.zeros((0,), dtype=torch.long)

    bus_vm_min = bus_x[:, BUS_VMIN_IDX]
    bus_vm_max = bus_x[:, BUS_VMAX_IDX]
    gen_pg_min = gen_x[:, GEN_PMIN_IDX]
    gen_pg_max = gen_x[:, GEN_PMAX_IDX]
    gen_qg_min = gen_x[:, GEN_QMIN_IDX]
    gen_qg_max = gen_x[:, GEN_QMAX_IDX]
    line_angmin = line_x[:, EDGE_ANGMIN_IDX]
    line_angmax = line_x[:, EDGE_ANGMAX_IDX]
    line_rate_a = line_x[:, EDGE_RATE_A_IDX]
    trafo_angmin = torch.zeros((0,), dtype=torch.float32)
    trafo_angmax = torch.zeros((0,), dtype=torch.float32)
    trafo_rate_a = torch.zeros((0,), dtype=torch.float32)

    bus_is_ref = torch.zeros(n_bus, dtype=torch.float32)
    for row in buses:
        if int(float(row.get("is_slack", 0) or 0)) == 1 or (row.get("bus_type") or "").lower() == "slack":
            bus_is_ref[int(row["node_index"])] = 1.0
    if not bool(bus_is_ref.any()):
        slack_idx = (bus_type_col == 3.0).nonzero(as_tuple=True)[0]
        if slack_idx.numel() > 0:
            bus_is_ref[slack_idx[0]] = 1.0
        else:
            bus_is_ref[0] = 1.0

    targets: Dict[str, torch.Tensor] = {}
    # Bus targets (rad, pu)
    targets["bus_va"] = torch.tensor(
        [math.radians(_f(r["label_theta_deg"])) for r in buses],
        dtype=torch.float32,
    )
    targets["bus_vm"] = torch.tensor([_f(r["Vm_pu"]) for r in buses], dtype=torch.float32)

    # Gen P from label / table (pu)
    pg_by_bus: Dict[int, float] = {}
    for row in gens:
        bid = bus_id_to_idx[str(int(float(row["bus_id"])))]
        pg_by_bus[bid] = _f(row["Pg_mw"]) / sbase
    targets["gen_pg"] = torch.tensor([pg_by_bus[gen_bus_list[j]] for j in range(n_gen)], dtype=torch.float32)
    targets["gen_qg"] = torch.zeros(n_gen, dtype=torch.float32)

    # Branch MW flow → pu (active power only; Q targets omitted in supervised sum if we skip keys — we add zeros)
    flow_pu: List[float] = []
    for row in branches:
        if int(float(row.get("line_enabled", 1) or 1)) != 1:
            continue
        flow_pu.append(_f(row["label_flow_mw"]) / sbase)
    if n_line > 0:
        fp = torch.tensor(flow_pu, dtype=torch.float32)
        targets["line_pf"] = fp
        targets["line_qf"] = torch.zeros(n_line, dtype=torch.float32)
        targets["line_pt"] = -fp
        targets["line_qt"] = torch.zeros(n_line, dtype=torch.float32)

    return GraphBatch(
        bus_x=bus_x,
        gen_x=gen_x,
        load_x=load_x,
        shunt_x=shunt_x,
        line_x=line_x,
        trafo_x=trafo_x,
        gen_bus=gen_bus,
        load_bus=load_bus,
        shunt_bus=shunt_bus,
        line_from=line_from,
        line_to=line_to,
        trafo_from=trafo_from,
        trafo_to=trafo_to,
        bus_vm_min=bus_vm_min,
        bus_vm_max=bus_vm_max,
        gen_pg_min=gen_pg_min,
        gen_pg_max=gen_pg_max,
        gen_qg_min=gen_qg_min,
        gen_qg_max=gen_qg_max,
        line_angmin=line_angmin,
        line_angmax=line_angmax,
        trafo_angmin=trafo_angmin,
        trafo_angmax=trafo_angmax,
        line_rate_a=line_rate_a,
        trafo_rate_a=trafo_rate_a,
        bus_is_ref=bus_is_ref,
        load_pd=load_x[:, LOAD_PD_IDX],
        load_qd=load_x[:, LOAD_QD_IDX],
        shunt_gs=shunt_x[:, SHUNT_GS_IDX] if n_shunt > 0 else torch.zeros(0, dtype=torch.float32),
        shunt_bs=shunt_x[:, SHUNT_BS_IDX] if n_shunt > 0 else torch.zeros(0, dtype=torch.float32),
        targets=targets,
    )


def discover_sharaf_sample_dirs(root: Path) -> List[Path]:
    """Return sorted directories under ``root`` that contain ``buses.csv``."""
    root = Path(root)
    out: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "buses.csv").is_file():
            out.append(p)
    return out


class SharafCSVDataset(Dataset):
    """One ``GraphBatch`` per sample folder (optionally repeat for longer training)."""

    def __init__(
        self,
        root: str | Path,
        *,
        sbase_mva: float = 100.0,
        repeat: int = 1,
        sample_dirs: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.sbase_mva = float(sbase_mva)
        self.repeat = max(1, int(repeat))
        if sample_dirs:
            self._dirs = [self.root / d for d in sample_dirs]
        else:
            self._dirs = discover_sharaf_sample_dirs(self.root)
        if not self._dirs:
            raise FileNotFoundError(f"No sample folders with buses.csv under {self.root}")

    def __len__(self) -> int:
        return len(self._dirs) * self.repeat

    def __getitem__(self, idx: int) -> GraphBatch:
        d = self._dirs[idx % len(self._dirs)]
        return graphbatch_from_sharaf_dir(d, sbase_mva=self.sbase_mva)
