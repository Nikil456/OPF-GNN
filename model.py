"""
HeteroGNN: Heterogeneous GNN for AC-OPF.
Uses SAGEConv per edge type with manual aggregation (avoids HeteroConv forward signature differences across PyG versions).
Type-specific MLP heads for bus (V_m, V_a) and generator (P_g, Q_g).
"""
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import HeteroData

# OPFDataset edge types (source, edge_type, target)
EDGE_TYPES = [
    ("bus", "ac_line", "bus"),
    ("bus", "transformer", "bus"),
    ("generator", "generator_link", "bus"),
    ("bus", "generator_link", "generator"),
    ("load", "load_link", "bus"),
    ("bus", "load_link", "load"),
    ("shunt", "shunt_link", "bus"),
    ("bus", "shunt_link", "shunt"),
]


def _edge_key(edge_type: tuple) -> str:
    return f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"


class HeteroGNN(nn.Module):
    """
    Physics-aware HeteroGNN: f(Grid Topology, Demands) -> Optimal State.
    Multi-task: bus -> (V_magnitude, V_angle), generator -> (P_g, Q_g).
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 3,
        out_bus: int = 2,
        out_generator: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_bus = out_bus
        self.out_generator = out_generator

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = nn.ModuleDict()
            for edge_type in EDGE_TYPES:
                conv_dict[_edge_key(edge_type)] = SAGEConv((-1, -1), hidden_channels)
            self.convs.append(conv_dict)

        self.bus_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_bus),
        )
        self.generator_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_generator),
        )

    def _hetero_forward(self, x_dict: dict, edge_index_dict: dict, conv_dict: nn.ModuleDict) -> dict:
        """Run one layer: for each edge type call SAGEConv((x_src, x_dst), edge_index), then aggregate by dst."""
        out_by_dst: dict[str, list[torch.Tensor]] = {}
        for edge_type in EDGE_TYPES:
            src, rel, dst = edge_type[0], edge_type[1], edge_type[2]
            if edge_type not in edge_index_dict or src not in x_dict or dst not in x_dict:
                continue
            edge_index = edge_index_dict[edge_type]
            if edge_index.numel() == 0:
                continue
            x_src, x_dst = x_dict[src], x_dict[dst]
            conv = conv_dict[_edge_key(edge_type)]
            # SAGEConv.forward(x, edge_index) with x = (x_src, x_dst) for bipartite
            h = conv((x_src, x_dst), edge_index)
            if dst not in out_by_dst:
                out_by_dst[dst] = []
            out_by_dst[dst].append(h)
        # aggregate by destination (sum)
        return {k: torch.stack(v, dim=0).sum(dim=0) for k, v in out_by_dst.items()}

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        edge_attr_dict: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        x_dict: node_type -> (N, F)
        edge_index_dict: (src, rel, dst) -> (2, E)
        edge_attr_dict: optional; not used in this baseline (could concatenate to edge in custom conv).
        """
        for conv_dict in self.convs:
            h_dict = self._hetero_forward(x_dict, edge_index_dict, conv_dict)
            x_dict = dict(x_dict)
            for k, h in h_dict.items():
                x_dict[k] = h.relu()

        out = {}
        if "bus" in x_dict:
            out["bus"] = self.bus_head(x_dict["bus"])
        if "generator" in x_dict:
            out["generator"] = self.generator_head(x_dict["generator"])
        return out

    @staticmethod
    def edge_index_dict_from_data(data: HeteroData) -> dict:
        """Build edge_index_dict from HeteroData (only edge types we use)."""
        edge_types = data.metadata()[1]
        edge_index_dict = {}
        for key in EDGE_TYPES:
            if key in edge_types and getattr(data[key], "edge_index", None) is not None:
                edge_index_dict[key] = data[key].edge_index
        return edge_index_dict

    @staticmethod
    def x_dict_from_data(data: HeteroData, node_types: tuple = ("bus", "generator", "load", "shunt")) -> dict:
        """Build x_dict from HeteroData. Use data.metadata()[0] so node types are found (HeteroData has no attr 'bus')."""
        x_dict = {}
        for nt in data.metadata()[0]:
            if nt not in node_types:
                continue
            try:
                x = data[nt].x
                if x is not None:
                    x_dict[nt] = x
            except (KeyError, AttributeError):
                pass
        return x_dict
