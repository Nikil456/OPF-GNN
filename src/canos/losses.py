from __future__ import annotations

from typing import Dict, Tuple

import torch

from .data import GraphBatch
from .utils import scatter_sum


def _ineq_violation(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return torch.relu(lo - x) + torch.relu(x - hi)


def _power_balance_mismatch(g: GraphBatch, pred: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    nb = g.bus_x.shape[0]

    inj_p = scatter_sum(pred["gen_pg"].unsqueeze(-1), g.gen_bus, nb).squeeze(-1)
    inj_q = scatter_sum(pred["gen_qg"].unsqueeze(-1), g.gen_bus, nb).squeeze(-1)

    load_p = scatter_sum(g.load_pd.unsqueeze(-1), g.load_bus, nb).squeeze(-1)
    load_q = scatter_sum(g.load_qd.unsqueeze(-1), g.load_bus, nb).squeeze(-1)

    shunt_p = scatter_sum((g.shunt_gs * (pred["bus_vm"][g.shunt_bus] ** 2)).unsqueeze(-1), g.shunt_bus, nb).squeeze(-1)
    shunt_q = scatter_sum((-g.shunt_bs * (pred["bus_vm"][g.shunt_bus] ** 2)).unsqueeze(-1), g.shunt_bus, nb).squeeze(-1)

    flow_p = (
        scatter_sum(pred["line_pf"].unsqueeze(-1), g.line_from, nb)
        + scatter_sum(pred["line_pt"].unsqueeze(-1), g.line_to, nb)
        + scatter_sum(pred["trafo_pf"].unsqueeze(-1), g.trafo_from, nb)
        + scatter_sum(pred["trafo_pt"].unsqueeze(-1), g.trafo_to, nb)
    ).squeeze(-1)

    flow_q = (
        scatter_sum(pred["line_qf"].unsqueeze(-1), g.line_from, nb)
        + scatter_sum(pred["line_qt"].unsqueeze(-1), g.line_to, nb)
        + scatter_sum(pred["trafo_qf"].unsqueeze(-1), g.trafo_from, nb)
        + scatter_sum(pred["trafo_qt"].unsqueeze(-1), g.trafo_to, nb)
    ).squeeze(-1)

    mismatch_p = inj_p - load_p - shunt_p - flow_p
    mismatch_q = inj_q - load_q - shunt_q - flow_q
    return mismatch_p, mismatch_q


def supervised_l2_loss(pred: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    loss = pred["bus_va"].new_tensor(0.0)
    for key in ["bus_va", "bus_vm", "gen_pg", "gen_qg"]:
        if key in targets:
            loss = loss + torch.mean((pred[key] - targets[key]) ** 2)

    for key in ["line_pf", "line_qf", "line_pt", "line_qt", "trafo_pf", "trafo_qf", "trafo_pt", "trafo_qt"]:
        if key in targets:
            loss = loss + torch.mean((pred[key] - targets[key]) ** 2)
    return loss


def constraint_loss(g: GraphBatch, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mismatch_p, mismatch_q = _power_balance_mismatch(g, pred)

    line_s_f = torch.sqrt(pred["line_pf"] ** 2 + pred["line_qf"] ** 2)
    line_s_t = torch.sqrt(pred["line_pt"] ** 2 + pred["line_qt"] ** 2)
    trafo_s_f = torch.sqrt(pred["trafo_pf"] ** 2 + pred["trafo_qf"] ** 2)
    trafo_s_t = torch.sqrt(pred["trafo_pt"] ** 2 + pred["trafo_qt"] ** 2)

    line_dtheta = pred["bus_va"][g.line_from] - pred["bus_va"][g.line_to]
    trafo_dtheta = pred["bus_va"][g.trafo_from] - pred["bus_va"][g.trafo_to]

    terms = {
        "ref_angle": torch.mean(torch.abs(pred["bus_va"][g.bus_is_ref > 0.5])),
        "power_balance_p": torch.mean(torch.abs(mismatch_p)),
        "power_balance_q": torch.mean(torch.abs(mismatch_q)),
        "line_thermal_f": torch.mean(torch.relu(line_s_f - g.line_rate_a)),
        "line_thermal_t": torch.mean(torch.relu(line_s_t - g.line_rate_a)),
        "trafo_thermal_f": torch.mean(torch.relu(trafo_s_f - g.trafo_rate_a)),
        "trafo_thermal_t": torch.mean(torch.relu(trafo_s_t - g.trafo_rate_a)),
        "line_angle": torch.mean(_ineq_violation(line_dtheta, g.line_angmin, g.line_angmax)),
        "trafo_angle": torch.mean(_ineq_violation(trafo_dtheta, g.trafo_angmin, g.trafo_angmax)),
        "bus_vm_bounds": torch.mean(_ineq_violation(pred["bus_vm"], g.bus_vm_min, g.bus_vm_max)),
        "gen_pg_bounds": torch.mean(_ineq_violation(pred["gen_pg"], g.gen_pg_min, g.gen_pg_max)),
        "gen_qg_bounds": torch.mean(_ineq_violation(pred["gen_qg"], g.gen_qg_min, g.gen_qg_max)),
    }
    terms["total"] = sum(terms.values())
    return terms


def compute_total_loss(
    g: GraphBatch,
    pred: Dict[str, torch.Tensor],
    constraint_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    supervised = supervised_l2_loss(pred, g.targets or {})
    constraints = constraint_loss(g, pred)
    total = supervised + constraint_weight * constraints["total"]
    return {
        "total": total,
        "supervised": supervised,
        "constraints_total": constraints["total"],
        **{f"constraint_{k}": v for k, v in constraints.items() if k != "total"},
    }
