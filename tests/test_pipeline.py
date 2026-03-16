"""
Smoke tests: model forward, loss, and transform without downloading OPFDataset.
Run from project root: pytest tests/ -v   or   python -m pytest tests/ -v
"""
import os
import sys

import torch

# run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import HeteroGNN, EDGE_TYPES
from losses import mse_loss, combined_loss
from torch_geometric.data import HeteroData


def _make_fake_hetero_data(num_bus=5, num_gen=2, num_load=3, num_shunt=1):
    """Minimal HeteroData with same node/edge structure as OPFDataset (no download)."""
    data = HeteroData()
    data["bus"].x = torch.randn(num_bus, 4)
    data["bus"].y = torch.randn(num_bus, 2)
    data["generator"].x = torch.randn(num_gen, 6)
    data["generator"].y = torch.randn(num_gen, 2)
    data["load"].x = torch.randn(num_load, 3)
    data["shunt"].x = torch.randn(num_shunt, 2)

    # minimal edges so each conv has something
    data["bus", "ac_line", "bus"].edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data["bus", "transformer", "bus"].edge_index = torch.empty(2, 0, dtype=torch.long)
    data["generator", "generator_link", "bus"].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data["bus", "generator_link", "generator"].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data["load", "load_link", "bus"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    data["bus", "load_link", "load"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    data["shunt", "shunt_link", "bus"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data["bus", "shunt_link", "shunt"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    return data


def test_model_forward():
    data = _make_fake_hetero_data()
    x_dict = HeteroGNN.x_dict_from_data(data)
    edge_index_dict = HeteroGNN.edge_index_dict_from_data(data)
    for key in EDGE_TYPES:
        if key not in edge_index_dict:
            edge_index_dict[key] = torch.empty(2, 0, dtype=torch.long)

    model = HeteroGNN(hidden_channels=8, num_layers=2, out_bus=2, out_generator=2)
    pred = model(x_dict, edge_index_dict)

    assert "bus" in pred and pred["bus"].shape == (5, 2)
    assert "generator" in pred and pred["generator"].shape == (2, 2)


def test_mse_loss():
    pred = {"bus": torch.randn(5, 2), "generator": torch.randn(2, 2)}
    target = {"bus": torch.randn(5, 2), "generator": torch.randn(2, 2)}
    loss = mse_loss(pred, target)
    assert loss.dim() == 0 and loss.item() >= 0


def test_combined_loss():
    pred = {"bus": torch.randn(5, 2), "generator": torch.randn(2, 2)}
    target = {"bus": torch.randn(5, 2), "generator": torch.randn(2, 2)}
    total, breakdown = combined_loss(pred, target, physics_weight=0.0)
    assert total.dim() == 0
    assert "mse" in breakdown and "physics" in breakdown
