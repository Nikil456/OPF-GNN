from __future__ import annotations

from dataclasses import dataclass
import inspect
import tarfile
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset


BUS_VMIN_IDX = 2
BUS_VMAX_IDX = 3
BUS_TYPE_IDX = 1

GEN_PMIN_IDX = 2
GEN_PMAX_IDX = 3
GEN_QMIN_IDX = 5
GEN_QMAX_IDX = 6

LOAD_PD_IDX = 0
LOAD_QD_IDX = 1

SHUNT_BS_IDX = 0
SHUNT_GS_IDX = 1

EDGE_ANGMIN_IDX = 0
EDGE_ANGMAX_IDX = 1
EDGE_BFR_IDX = 2
EDGE_BTO_IDX = 3
EDGE_BR_R_IDX = 4
EDGE_BR_X_IDX = 5
EDGE_RATE_A_IDX = 6
TRAFO_TAP_IDX = 9
TRAFO_SHIFT_IDX = 10

BUS_Y_VA_IDX = 0
BUS_Y_VM_IDX = 1
GEN_Y_PG_IDX = 0
GEN_Y_QG_IDX = 1
EDGE_Y_PT_IDX = 0
EDGE_Y_QT_IDX = 1
EDGE_Y_PF_IDX = 2
EDGE_Y_QF_IDX = 3


@dataclass
class GraphBatch:
    bus_x: torch.Tensor
    gen_x: torch.Tensor
    load_x: torch.Tensor
    shunt_x: torch.Tensor
    line_x: torch.Tensor
    trafo_x: torch.Tensor

    gen_bus: torch.Tensor
    load_bus: torch.Tensor
    shunt_bus: torch.Tensor
    line_from: torch.Tensor
    line_to: torch.Tensor
    trafo_from: torch.Tensor
    trafo_to: torch.Tensor

    bus_vm_min: torch.Tensor
    bus_vm_max: torch.Tensor
    gen_pg_min: torch.Tensor
    gen_pg_max: torch.Tensor
    gen_qg_min: torch.Tensor
    gen_qg_max: torch.Tensor
    line_angmin: torch.Tensor
    line_angmax: torch.Tensor
    trafo_angmin: torch.Tensor
    trafo_angmax: torch.Tensor
    line_rate_a: torch.Tensor
    trafo_rate_a: torch.Tensor
    bus_is_ref: torch.Tensor

    load_pd: torch.Tensor
    load_qd: torch.Tensor
    shunt_gs: torch.Tensor
    shunt_bs: torch.Tensor

    targets: Optional[Dict[str, torch.Tensor]] = None

    def to(self, device: torch.device) -> "GraphBatch":
        kwargs = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(device)
            elif isinstance(v, dict):
                kwargs[k] = {dk: dv.to(device) for dk, dv in v.items()}
            else:
                kwargs[k] = v
        return GraphBatch(**kwargs)


def _infer_reference_bus_mask(bus_x: torch.Tensor) -> torch.Tensor:
    bus_type = bus_x[:, BUS_TYPE_IDX]
    ref_mask = torch.isclose(bus_type, bus_type.new_tensor(3.0))
    if not torch.any(ref_mask):
        ref_mask = torch.zeros_like(bus_type)
        ref_mask[0] = 1.0
    else:
        ref_mask = ref_mask.to(bus_x.dtype)
    return ref_mask


def graphbatch_from_pyg(data: Any) -> GraphBatch:
    bus_x = data["bus"].x.float()
    gen_x = data["generator"].x.float()
    load_x = data["load"].x.float()
    shunt_x = data["shunt"].x.float()

    line_store = data["bus", "ac_line", "bus"]
    trafo_store = data["bus", "transformer", "bus"]
    gen_link = data["generator", "generator_link", "bus"].edge_index
    load_link = data["load", "load_link", "bus"].edge_index
    shunt_link = data["shunt", "shunt_link", "bus"].edge_index

    line_x = line_store.edge_attr.float()
    trafo_x = trafo_store.edge_attr.float()

    line_targets = getattr(line_store, "edge_label", None)
    trafo_targets = getattr(trafo_store, "edge_label", None)
    bus_targets = getattr(data["bus"], "y", None)
    gen_targets = getattr(data["generator"], "y", None)

    targets: Dict[str, torch.Tensor] = {}
    if bus_targets is not None:
        bus_targets = bus_targets.float()
        targets["bus_va"] = bus_targets[:, BUS_Y_VA_IDX]
        targets["bus_vm"] = bus_targets[:, BUS_Y_VM_IDX]
    if gen_targets is not None:
        gen_targets = gen_targets.float()
        targets["gen_pg"] = gen_targets[:, GEN_Y_PG_IDX]
        targets["gen_qg"] = gen_targets[:, GEN_Y_QG_IDX]
    if line_targets is not None:
        line_targets = line_targets.float()
        targets["line_pt"] = line_targets[:, EDGE_Y_PT_IDX]
        targets["line_qt"] = line_targets[:, EDGE_Y_QT_IDX]
        targets["line_pf"] = line_targets[:, EDGE_Y_PF_IDX]
        targets["line_qf"] = line_targets[:, EDGE_Y_QF_IDX]
    if trafo_targets is not None:
        trafo_targets = trafo_targets.float()
        targets["trafo_pt"] = trafo_targets[:, EDGE_Y_PT_IDX]
        targets["trafo_qt"] = trafo_targets[:, EDGE_Y_QT_IDX]
        targets["trafo_pf"] = trafo_targets[:, EDGE_Y_PF_IDX]
        targets["trafo_qf"] = trafo_targets[:, EDGE_Y_QF_IDX]

    return GraphBatch(
        bus_x=bus_x,
        gen_x=gen_x,
        load_x=load_x,
        shunt_x=shunt_x,
        line_x=line_x,
        trafo_x=trafo_x,
        gen_bus=gen_link[1].long(),
        load_bus=load_link[1].long(),
        shunt_bus=shunt_link[1].long(),
        line_from=line_store.edge_index[0].long(),
        line_to=line_store.edge_index[1].long(),
        trafo_from=trafo_store.edge_index[0].long(),
        trafo_to=trafo_store.edge_index[1].long(),
        bus_vm_min=bus_x[:, BUS_VMIN_IDX],
        bus_vm_max=bus_x[:, BUS_VMAX_IDX],
        gen_pg_min=gen_x[:, GEN_PMIN_IDX],
        gen_pg_max=gen_x[:, GEN_PMAX_IDX],
        gen_qg_min=gen_x[:, GEN_QMIN_IDX],
        gen_qg_max=gen_x[:, GEN_QMAX_IDX],
        line_angmin=line_x[:, EDGE_ANGMIN_IDX],
        line_angmax=line_x[:, EDGE_ANGMAX_IDX],
        trafo_angmin=trafo_x[:, EDGE_ANGMIN_IDX],
        trafo_angmax=trafo_x[:, EDGE_ANGMAX_IDX],
        line_rate_a=line_x[:, EDGE_RATE_A_IDX],
        trafo_rate_a=trafo_x[:, EDGE_RATE_A_IDX],
        bus_is_ref=_infer_reference_bus_mask(bus_x),
        load_pd=load_x[:, LOAD_PD_IDX],
        load_qd=load_x[:, LOAD_QD_IDX],
        shunt_gs=shunt_x[:, SHUNT_GS_IDX],
        shunt_bs=shunt_x[:, SHUNT_BS_IDX],
        targets=targets or None,
    )


def patch_torch_geometric_tar_extract_compat() -> None:
    """Make PyG OPFDataset download/extraction work on Python < 3.12.

    Recent PyG versions call `TarFile.extractall(..., filter='data')`, but the
    `filter` keyword is not available in older Python versions such as 3.10.
    This patch drops the unsupported keyword while preserving default behavior.
    """

    signature = inspect.signature(tarfile.TarFile.extractall)
    if "filter" in signature.parameters:
        return

    original_extractall = tarfile.TarFile.extractall

    if getattr(original_extractall, "_canos_patched", False):
        return

    def _extractall_compat(self, path=".", members=None, *, numeric_owner=False, filter=None):
        del filter
        return original_extractall(self, path=path, members=members, numeric_owner=numeric_owner)

    _extractall_compat._canos_patched = True  # type: ignore[attr-defined]
    tarfile.TarFile.extractall = _extractall_compat


class SyntheticPowerDataset(Dataset):
    """Synthetic dataset for smoke-testing the training pipeline.

    This is not a physics-accurate OPF dataset; it only provides tensors with
    correct shapes and rough value ranges so the model and losses can run.
    """

    def __init__(
        self,
        size: int = 1024,
        n_bus: int = 64,
        n_gen: int = 24,
        n_load: int = 40,
        n_shunt: int = 16,
        n_line: int = 96,
        n_trafo: int = 24,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.n_bus = n_bus
        self.n_gen = n_gen
        self.n_load = n_load
        self.n_shunt = n_shunt
        self.n_line = n_line
        self.n_trafo = n_trafo
        self.rng = torch.Generator().manual_seed(seed)

        self.bus_dim = 4
        self.gen_dim = 11
        self.load_dim = 2
        self.shunt_dim = 2
        self.line_dim = 9
        self.trafo_dim = 11

    def __len__(self) -> int:
        return self.size

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.rng)

    def __getitem__(self, idx: int) -> GraphBatch:
        del idx
        nb = self.n_bus
        ng = self.n_gen
        nl = self.n_load
        ns = self.n_shunt
        ne = self.n_line
        nt = self.n_trafo

        bus_x = self._rand(nb, self.bus_dim)
        gen_x = self._rand(ng, self.gen_dim)
        load_x = self._rand(nl, self.load_dim)
        shunt_x = self._rand(ns, self.shunt_dim)
        line_x = self._rand(ne, self.line_dim)
        trafo_x = self._rand(nt, self.trafo_dim)

        gen_bus = torch.randint(0, nb, (ng,), generator=self.rng)
        load_bus = torch.randint(0, nb, (nl,), generator=self.rng)
        shunt_bus = torch.randint(0, nb, (ns,), generator=self.rng)

        line_from = torch.randint(0, nb, (ne,), generator=self.rng)
        line_to = torch.randint(0, nb, (ne,), generator=self.rng)
        trafo_from = torch.randint(0, nb, (nt,), generator=self.rng)
        trafo_to = torch.randint(0, nb, (nt,), generator=self.rng)

        bus_vm_min = 0.9 * torch.ones(nb)
        bus_vm_max = 1.1 * torch.ones(nb)
        gen_pg_min = torch.zeros(ng)
        gen_pg_max = 2.0 * torch.ones(ng)
        gen_qg_min = -0.8 * torch.ones(ng)
        gen_qg_max = 0.8 * torch.ones(ng)

        line_angmin = -0.6 * torch.ones(ne)
        line_angmax = 0.6 * torch.ones(ne)
        trafo_angmin = -0.6 * torch.ones(nt)
        trafo_angmax = 0.6 * torch.ones(nt)

        line_rate_a = 2.0 * torch.ones(ne)
        trafo_rate_a = 2.5 * torch.ones(nt)

        bus_is_ref = torch.zeros(nb)
        bus_is_ref[0] = 1.0

        load_pd = 0.2 + 0.8 * self._rand(nl)
        load_qd = 0.1 + 0.5 * self._rand(nl)
        shunt_bs = 0.01 * self._rand(ns)
        shunt_gs = 0.01 * self._rand(ns)

        targets = {
            "bus_va": -0.2 + 0.4 * self._rand(nb),
            "bus_vm": 0.95 + 0.1 * self._rand(nb),
            "gen_pg": 0.2 + 1.6 * self._rand(ng),
            "gen_qg": -0.5 + 1.0 * self._rand(ng),
        }

        return GraphBatch(
            bus_x=bus_x,
            gen_x=gen_x,
            load_x=load_x,
            shunt_x=torch.stack([shunt_bs, shunt_gs], dim=-1),
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
            load_pd=load_pd,
            load_qd=load_qd,
            shunt_bs=shunt_bs,
            shunt_gs=shunt_gs,
            targets=targets,
        )
