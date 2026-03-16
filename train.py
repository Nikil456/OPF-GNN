"""
OPF-GNN: Deep Learning for AC-OPF using PyTorch Geometric.
Load data, print summary, then train the model.
"""
import argparse
import os

import torch
from torch_geometric.datasets import OPFDataset

from model import HeteroGNN, EDGE_TYPES
from losses import mse_loss


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_sample(data, device, fill_missing_edges=True):
    """Build x_dict, edge_index_dict, target_dict and move to device."""
    x_dict = HeteroGNN.x_dict_from_data(data)
    edge_index_dict = HeteroGNN.edge_index_dict_from_data(data)
    if fill_missing_edges:
        for key in EDGE_TYPES:
            if key not in edge_index_dict:
                edge_index_dict[key] = torch.empty(2, 0, dtype=torch.long)
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    target_dict = {
        "bus": data["bus"].y.to(device),
        "generator": data["generator"].y.to(device),
    }
    return x_dict, edge_index_dict, target_dict


def main():
    parser = argparse.ArgumentParser(description="Train OPF-GNN on case14.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channels")
    parser.add_argument("--layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--num_groups", type=int, default=1, help="OPFDataset num_groups (1–20)")
    parser.add_argument("--no_summary", action="store_true", help="Skip printing data summary")
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(__file__), "data", "OPF")
    device = get_device()
    print(f"Device: {device}")

    train_dataset = OPFDataset(
        root=root,
        split="train",
        case_name="pglib_opf_case14_ieee",
        num_groups=args.num_groups,
    )
    data = train_dataset[0]

    if not args.no_summary:
        print("=" * 60)
        print("OPF-GNN — HeteroData Summary (pglib_opf_case14_ieee)")
        print("=" * 60)
        print(f"Dataset size: {len(train_dataset)} samples")
        for node_type in ["bus", "generator", "load", "shunt"]:
            if hasattr(data, node_type) and data[node_type] is not None:
                n = data[node_type]
                x, y = getattr(n, "x", None), getattr(n, "y", None)
                print(f"  [{node_type}] x: {x.shape if x is not None else None}, y: {y.shape if y is not None else None}")
        print("=" * 60)

    model = HeteroGNN(
        hidden_channels=args.hidden,
        num_layers=args.layers,
        out_bus=2,
        out_generator=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            x_dict, edge_index_dict, target_dict = prepare_sample(sample, device)
            optimizer.zero_grad()
            pred = model(x_dict, edge_index_dict)
            loss = mse_loss(pred, target_dict)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        mean_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch:3d}  train loss: {mean_loss:.6f}")

    print("Done.")


if __name__ == "__main__":
    main()
