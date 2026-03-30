"""
Heterogeneous OPF graph sketch: topology before training; after training, bus/generator
nodes are colored by mean absolute prediction error on a reference sample.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np

from model import EDGE_TYPES

if TYPE_CHECKING:
    import torch
    from torch_geometric.data import HeteroData


NODE_DRAW_ORDER = ("bus", "generator", "load", "shunt")
NODE_FACE = {
    "bus": "#3b82f6",
    "generator": "#22c55e",
    "load": "#f97316",
    "shunt": "#a855f7",
}
NODE_SIZE = {"bus": 280, "generator": 220, "load": 180, "shunt": 160}
EDGE_STYLE = {
    "ac_line": {"color": "#64748b", "linewidth": 1.4, "linestyle": "-"},
    "transformer": {"color": "#0d9488", "linewidth": 1.6, "linestyle": "--"},
    "generator_link": {"color": "#ca8a04", "linewidth": 1.0, "linestyle": ":"},
    "load_link": {"color": "#c2410c", "linewidth": 1.0, "linestyle": "-."},
    "shunt_link": {"color": "#7c3aed", "linewidth": 1.0, "linestyle": ":"},
}


def hetero_data_to_graph(data: HeteroData) -> nx.Graph:
    """Flatten HeteroData into an undirected NetworkX graph with typed node keys (type, index)."""
    G = nx.Graph()
    for nt in data.metadata()[0]:
        try:
            store = data[nt]
            n = int(store.num_nodes)
        except (KeyError, AttributeError):
            continue
        for i in range(n):
            G.add_node((nt, i), node_type=nt)

    edge_types = set(data.metadata()[1])
    for et in EDGE_TYPES:
        if et not in edge_types:
            continue
        ei = data[et].edge_index
        if ei is None or ei.numel() == 0:
            continue
        src_t, rel, dst_t = et[0], et[1], et[2]
        arr = ei.cpu().numpy()
        for k in range(arr.shape[1]):
            u = (src_t, int(arr[0, k]))
            v = (dst_t, int(arr[1, k]))
            if not G.has_node(u) or not G.has_node(v):
                continue
            if G.has_edge(u, v):
                continue
            G.add_edge(u, v, edge_type=rel)
    return G


def _layout(G: nx.Graph, seed: int = 42) -> dict:
    n = max(len(G), 1)
    return nx.spring_layout(G, seed=seed, k=2.0 / np.sqrt(n), iterations=50)


def save_network_sketch(
    data: HeteroData,
    path: str,
    phase: Literal["before", "after"],
    pred_dict: dict[str, "torch.Tensor"] | None = None,
    target_dict: dict[str, "torch.Tensor"] | None = None,
    seed: int = 42,
) -> None:
    """
    Save a PNG sketch. phase=\"before\": uniform colors by node type (topology only).
    phase=\"after\": bus and generator nodes colored by mean |pred - target| (requires pred/target).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = hetero_data_to_graph(data)
    if len(G.nodes) == 0:
        raise ValueError("HeteroData has no nodes to sketch")

    pos = _layout(G, seed=seed)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor("#f8fafc")

    for u, v, ed in G.edges(data=True):
        rel = ed.get("edge_type", "")
        st = EDGE_STYLE.get(rel, {"color": "#94a3b8", "linewidth": 1.0, "linestyle": "-"})
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=st["color"],
            linewidth=st["linewidth"],
            linestyle=st["linestyle"],
            alpha=0.85,
            zorder=1,
        )

    if phase == "before":
        ax.set_title(
            "OPF heterogeneous graph — Before training\n"
            "(topology only; colors = node type)",
            fontsize=12,
        )
        for nt in NODE_DRAW_ORDER:
            nodes = [n for n in G.nodes if n[0] == nt]
            if not nodes:
                continue
            xs = [pos[n][0] for n in nodes]
            ys = [pos[n][1] for n in nodes]
            ax.scatter(
                xs,
                ys,
                s=NODE_SIZE.get(nt, 200),
                c=NODE_FACE[nt],
                label=nt,
                zorder=2,
                edgecolors="#1e293b",
                linewidths=0.8,
            )
        from matplotlib.lines import Line2D

        leg_nodes = ax.legend(loc="upper left", fontsize=9, framealpha=0.92, title="node types")
        ax.add_artist(leg_nodes)
        legend_elems = [
            Line2D([0], [0], color=EDGE_STYLE["ac_line"]["color"], lw=2, label="ac_line"),
            Line2D([0], [0], color=EDGE_STYLE["transformer"]["color"], lw=2, linestyle="--", label="transformer"),
            Line2D([0], [0], color=EDGE_STYLE["generator_link"]["color"], lw=2, linestyle=":", label="generator_link"),
            Line2D([0], [0], color=EDGE_STYLE["load_link"]["color"], lw=2, linestyle="-.", label="load_link"),
            Line2D([0], [0], color=EDGE_STYLE["shunt_link"]["color"], lw=2, linestyle=":", label="shunt_link"),
        ]
        ax.legend(handles=legend_elems, loc="lower left", fontsize=8, title="edge types", framealpha=0.92)

    else:
        if pred_dict is None or target_dict is None:
            raise ValueError('phase="after" requires pred_dict and target_dict')
        bus_err: dict[int, float] = {}
        gen_err: dict[int, float] = {}
        if "bus" in pred_dict and "bus" in target_dict:
            d = (pred_dict["bus"] - target_dict["bus"]).abs().mean(dim=-1).detach().cpu().numpy()
            for i, e in enumerate(d):
                bus_err[i] = float(e)
        if "generator" in pred_dict and "generator" in target_dict:
            d = (pred_dict["generator"] - target_dict["generator"]).abs().mean(dim=-1).detach().cpu().numpy()
            for i, e in enumerate(d):
                gen_err[i] = float(e)

        err_values = list(bus_err.values()) + list(gen_err.values())
        if err_values:
            vmax = max(err_values) * 1.05 + 1e-8
            vmin = 0.0
        else:
            vmin, vmax = 0.0, 1.0

        cmap = plt.colormaps["YlOrRd"]
        mappable = None
        from matplotlib.lines import Line2D

        for nt in NODE_DRAW_ORDER:
            nodes = [n for n in G.nodes if n[0] == nt]
            if not nodes:
                continue
            xs = [pos[n][0] for n in nodes]
            ys = [pos[n][1] for n in nodes]
            if nt == "bus":
                cs = [bus_err.get(n[1], 0.0) for n in nodes]
                mappable = ax.scatter(
                    xs,
                    ys,
                    s=NODE_SIZE["bus"],
                    c=cs,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    zorder=2,
                    edgecolors="#1e293b",
                    linewidths=0.8,
                )
            elif nt == "generator":
                cs = [gen_err.get(n[1], 0.0) for n in nodes]
                sc_g = ax.scatter(
                    xs,
                    ys,
                    s=NODE_SIZE["generator"],
                    c=cs,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    zorder=2,
                    edgecolors="#1e293b",
                    linewidths=0.8,
                )
                if mappable is None:
                    mappable = sc_g
            else:
                ax.scatter(
                    xs,
                    ys,
                    s=NODE_SIZE.get(nt, 180),
                    c="#94a3b8",
                    zorder=2,
                    edgecolors="#1e293b",
                    linewidths=0.8,
                )
        if mappable is None:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            mappable = ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
            mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("mean |pred − target|", fontsize=10)
        ax.set_title(
            "OPF heterogeneous graph — After training\n"
            "(bus & generator fill = error on reference sample; gray loads/shunts = not predicted)",
            fontsize=12,
        )
        # Custom legend: avoid matplotlib picking one colormap swatch per scatter (misleading).
        after_handles = []
        if any(n[0] == "bus" for n in G.nodes):
            after_handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="none",
                    markerfacecolor="none",
                    markeredgecolor="#1e293b",
                    markersize=11,
                    markeredgewidth=1.5,
                    linestyle="none",
                    label="Bus — interior fill ↔ colorbar (error)",
                )
            )
        if any(n[0] == "generator" for n in G.nodes):
            after_handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="none",
                    markerfacecolor="none",
                    markeredgecolor="#1e293b",
                    markersize=9,
                    markeredgewidth=1.5,
                    linestyle="none",
                    label="Generator — interior fill ↔ colorbar (error)",
                )
            )
        if any(n[0] == "load" for n in G.nodes):
            after_handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="none",
                    markerfacecolor="#94a3b8",
                    markeredgecolor="#1e293b",
                    markersize=8,
                    markeredgewidth=0.8,
                    linestyle="none",
                    label="Load — fixed gray (no ŷ in model)",
                )
            )
        if any(n[0] == "shunt" for n in G.nodes):
            after_handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="none",
                    markerfacecolor="#94a3b8",
                    markeredgecolor="#1e293b",
                    markersize=8,
                    markeredgewidth=0.8,
                    linestyle="none",
                    label="Shunt — fixed gray (no ŷ in model)",
                )
            )
        nodetype_leg = ax.legend(
            handles=after_handles,
            loc="upper left",
            fontsize=8,
            framealpha=0.95,
            title="Node markers",
        )
        ax.add_artist(nodetype_leg)
        edge_legend_elems = [
            Line2D([0], [0], color=EDGE_STYLE["ac_line"]["color"], lw=2, label="ac_line"),
            Line2D([0], [0], color=EDGE_STYLE["transformer"]["color"], lw=2, linestyle="--", label="transformer"),
            Line2D([0], [0], color=EDGE_STYLE["generator_link"]["color"], lw=2, linestyle=":", label="generator_link"),
            Line2D([0], [0], color=EDGE_STYLE["load_link"]["color"], lw=2, linestyle="-.", label="load_link"),
            Line2D([0], [0], color=EDGE_STYLE["shunt_link"]["color"], lw=2, linestyle=":", label="shunt_link"),
        ]
        ax.legend(
            handles=edge_legend_elems,
            loc="lower left",
            fontsize=8,
            title="Edge types",
            framealpha=0.95,
        )

    ax.axis("off")
    fig.text(
        0.5,
        0.02,
        f"Image generated {phase} training · same graph layout (seed={seed}) in both figures",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
