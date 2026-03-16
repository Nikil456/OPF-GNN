"""
Loss: MSE for regression + placeholder for Physics-Informed Loss.
Physics: AC power flow S = V · I^* (complex power balance); can add penalty terms later.
"""
import torch
import torch.nn as nn


def mse_loss(
    pred_dict: dict,
    target_dict: dict,
    node_types: tuple = ("bus", "generator"),
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Multi-task MSE over node types. pred_dict/target_dict: {node_type: tensor}.
    """
    losses = []
    for nt in node_types:
        if nt not in pred_dict or nt not in target_dict:
            continue
        p, t = pred_dict[nt], target_dict[nt]
        if p is None or t is None:
            continue
        losses.append(nn.functional.mse_loss(p, t, reduction=reduction))
    if not losses:
        device = None
        if pred_dict:
            device = next(iter(pred_dict.values())).device
        elif target_dict:
            device = next(iter(target_dict.values())).device
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def physics_informed_loss_placeholder(
    data,
    pred_bus_y: torch.Tensor,
    pred_gen_y: torch.Tensor,
    bus_idx: torch.Tensor,
    gen_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Placeholder for Physics-Informed Loss.
    Penalize violation of power flow equations: S = V · I^* (complex power balance).
    Can be extended with:
      - Power balance at each bus: P_in - P_out = P_demand, Q similarly.
      - Line flow limits; voltage bounds.
    Returns zero for now so training can use only MSE.
    """
    # TODO: implement power balance residual from (V, I) or (P, Q) and add penalty.
    del data, pred_bus_y, pred_gen_y, bus_idx, gen_idx
    return torch.tensor(0.0)


def combined_loss(
    pred_dict: dict,
    target_dict: dict,
    data=None,
    physics_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    combined_loss = mse_loss + physics_weight * physics_informed_loss_placeholder(...).
    Returns (total_loss, {"mse": ..., "physics": ...}).
    """
    mse = mse_loss(pred_dict, target_dict)
    physics = torch.tensor(0.0, device=mse.device)
    if physics_weight > 0 and data is not None and "bus" in pred_dict and "generator" in pred_dict:
        physics = physics_informed_loss_placeholder(
            data,
            pred_dict["bus"],
            pred_dict["generator"],
            bus_idx=None,
            gen_idx=None,
        ).to(mse.device)
    total = mse + physics_weight * physics
    return total, {"mse": mse.detach(), "physics": physics.detach()}
