from __future__ import annotations

import torch


EPS = 1e-9


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.shape[-1]))
    out.index_add_(0, index, src)
    return out


def bounded_sigmoid(raw: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(raw) * (upper - lower) + lower


def pairwise_complex_voltage(vm: torch.Tensor, va: torch.Tensor) -> torch.Tensor:
    return torch.polar(vm, va)


def complex_admittance_from_impedance(r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    z = torch.complex(r, x)
    return 1.0 / (z + (EPS + 0.0j))
