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
from network_sketch import save_network_sketch


def save_scalar_loss_plot(epochs: list[int], losses: list[float], path: str) -> None:
    """Save mean training MSE per epoch (scalar loss) to a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, color="#2563eb", marker="o", markersize=4, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean train MSE")
    ax.set_title("OPF-GNN scalar training loss")
    ax.grid(True, alpha=0.35)
    ax.set_xlim(left=0.5, right=max(epochs, default=1) + 0.5)
    fig.tight_layout()
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    default_plot = os.path.join(os.path.dirname(__file__), "outputs", "scalar_loss.png")
    parser.add_argument(
        "--plot_path",
        default=default_plot,
        help="Path for scalar loss curve PNG (default: outputs/scalar_loss.png under project root)",
    )
    parser.add_argument("--no_plot", action="store_true", help="Do not save the loss plot")
    _out = os.path.join(os.path.dirname(__file__), "outputs")
    parser.add_argument(
        "--network_sketch_before",
        default=os.path.join(_out, "network_sketch_before_training.png"),
        help="PNG path for topology sketch (saved before training)",
    )
    parser.add_argument(
        "--network_sketch_after",
        default=os.path.join(_out, "network_sketch_after_training.png"),
        help="PNG path for sketch after training (node color = prediction error)",
    )
    parser.add_argument("--no_network_sketch", action="store_true", help="Do not save network sketches")
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

    sketch_sample = train_dataset[0]
    if not args.no_network_sketch:
        save_network_sketch(sketch_sample, args.network_sketch_before, phase="before")
        print(f"Saved network sketch (before training): {args.network_sketch_before}")

    model = HeteroGNN(
        hidden_channels=args.hidden,
        num_layers=args.layers,
        out_bus=2,
        out_generator=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_losses: list[float] = []
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
        epoch_losses.append(mean_loss)
        print(f"Epoch {epoch:3d}  train loss: {mean_loss:.6f}")

    if not args.no_plot and epoch_losses:
        epochs_x = list(range(1, len(epoch_losses) + 1))
        save_scalar_loss_plot(epochs_x, epoch_losses, args.plot_path)
        print(f"Saved scalar loss plot: {args.plot_path}")

    if not args.no_network_sketch:
        model.eval()
        with torch.no_grad():
            x_dict, edge_index_dict, target_dict = prepare_sample(sketch_sample, device)
            pred_dict = model(x_dict, edge_index_dict)
        save_network_sketch(
            sketch_sample,
            args.network_sketch_after,
            phase="after",
            pred_dict=pred_dict,
            target_dict=target_dict,
        )
        print(f"Saved network sketch (after training): {args.network_sketch_after}")

    print("Done.")


if __name__ == "__main__":
    main()
