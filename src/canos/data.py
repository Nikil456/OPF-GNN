from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


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
TRAFO_TAP_IDX = 7
TRAFO_SHIFT_IDX = 8


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
