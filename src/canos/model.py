from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .data import GraphBatch
from .utils import (
    bounded_sigmoid,
    complex_admittance_from_impedance,
    pairwise_complex_voltage,
    scatter_sum,
)


LINE_BFR_IDX = 2
LINE_BTO_IDX = 3
LINE_R_IDX = 4
LINE_X_IDX = 5
TRAFO_BFR_IDX = 2
TRAFO_BTO_IDX = 3
TRAFO_R_IDX = 4
TRAFO_X_IDX = 5
TRAFO_TAP_IDX = 9
TRAFO_SHIFT_IDX = 10


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CANOS(nn.Module):
    """Constraint-Augmented Neural OPF Solver (CANOS) style architecture.

    This is a faithful reproduction of core ideas from the paper:
    - heterogeneous encode-process-decode GNN
    - typed message passing with residual updates
    - bounded outputs for generator and voltage magnitudes
    - branch flow derivation from voltage and generator outputs
    """

    def __init__(
        self,
        bus_in: int,
        gen_in: int,
        load_in: int,
        shunt_in: int,
        line_in: int,
        trafo_in: int,
        hidden_size: int = 128,
        num_message_passing_steps: int = 48,
        decoder_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_steps = num_message_passing_steps

        self.enc_bus = nn.Linear(bus_in, hidden_size)
        self.enc_gen = nn.Linear(gen_in, hidden_size)
        self.enc_load = nn.Linear(load_in, hidden_size)
        self.enc_shunt = nn.Linear(shunt_in, hidden_size)
        self.enc_line = nn.Linear(line_in, hidden_size)
        self.enc_trafo = nn.Linear(trafo_in, hidden_size)

        self.line_edge_upd = nn.ModuleList(
            [MLP(3 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )
        self.trafo_edge_upd = nn.ModuleList(
            [MLP(3 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )

        self.msg_bus_from_line = nn.ModuleList(
            [MLP(2 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )
        self.msg_bus_from_trafo = nn.ModuleList(
            [MLP(2 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )
        self.msg_bus_from_gen = nn.ModuleList(
            [MLP(2 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )
        self.msg_bus_from_load = nn.ModuleList(
            [MLP(2 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )
        self.msg_bus_from_shunt = nn.ModuleList(
            [MLP(2 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )

        self.msg_gen_from_bus = nn.ModuleList(
            [MLP(2 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )
        self.msg_load_from_bus = nn.ModuleList(
            [MLP(2 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )
        self.msg_shunt_from_bus = nn.ModuleList(
            [MLP(2 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )

        self.bus_upd = nn.ModuleList(
            [MLP(2 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )
        self.gen_upd = nn.ModuleList(
            [MLP(2 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )
        self.load_upd = nn.ModuleList(
            [MLP(2 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )
        self.shunt_upd = nn.ModuleList(
            [MLP(2 * hidden_size, hidden_size, hidden_size) for _ in range(num_message_passing_steps)]
        )

        self.bus_decoder = nn.Sequential(
            MLP(hidden_size, decoder_hidden, decoder_hidden),
            nn.Linear(decoder_hidden, 2),
        )
        self.gen_decoder = nn.Sequential(
            MLP(hidden_size, decoder_hidden, decoder_hidden),
            nn.Linear(decoder_hidden, 2),
        )

    def _process(self, g: GraphBatch, h: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for t in range(self.num_steps):
            h_line_new = self.line_edge_upd[t](
                torch.cat([h["bus"][g.line_from], h["bus"][g.line_to], h["line"]], dim=-1)
            )
            h_trafo_new = self.trafo_edge_upd[t](
                torch.cat([h["bus"][g.trafo_from], h["bus"][g.trafo_to], h["trafo"]], dim=-1)
            )

            h["line"] = h["line"] + h_line_new
            h["trafo"] = h["trafo"] + h_trafo_new

            line_msg_fwd = self.msg_bus_from_line[t](torch.cat([h["bus"][g.line_from], h["line"]], dim=-1))
            line_msg_rev = self.msg_bus_from_line[t](torch.cat([h["bus"][g.line_to], h["line"]], dim=-1))
            trafo_msg_fwd = self.msg_bus_from_trafo[t](torch.cat([h["bus"][g.trafo_from], h["trafo"]], dim=-1))
            trafo_msg_rev = self.msg_bus_from_trafo[t](torch.cat([h["bus"][g.trafo_to], h["trafo"]], dim=-1))

            bus_from_lines = scatter_sum(line_msg_fwd, g.line_to, h["bus"].shape[0]) + scatter_sum(
                line_msg_rev, g.line_from, h["bus"].shape[0]
            )
            bus_from_trafos = scatter_sum(trafo_msg_fwd, g.trafo_to, h["bus"].shape[0]) + scatter_sum(
                trafo_msg_rev, g.trafo_from, h["bus"].shape[0]
            )

            bus_from_gen = scatter_sum(
                self.msg_bus_from_gen[t](torch.cat([h["gen"], h["bus"][g.gen_bus]], dim=-1)),
                g.gen_bus,
                h["bus"].shape[0],
            )
            bus_from_load = scatter_sum(
                self.msg_bus_from_load[t](torch.cat([h["load"], h["bus"][g.load_bus]], dim=-1)),
                g.load_bus,
                h["bus"].shape[0],
            )
            bus_from_shunt = scatter_sum(
                self.msg_bus_from_shunt[t](torch.cat([h["shunt"], h["bus"][g.shunt_bus]], dim=-1)),
                g.shunt_bus,
                h["bus"].shape[0],
            )

            bus_agg = bus_from_lines + bus_from_trafos + bus_from_gen + bus_from_load + bus_from_shunt
            gen_agg = self.msg_gen_from_bus[t](torch.cat([h["bus"][g.gen_bus], h["gen"]], dim=-1))
            load_agg = self.msg_load_from_bus[t](torch.cat([h["bus"][g.load_bus], h["load"]], dim=-1))
            shunt_agg = self.msg_shunt_from_bus[t](torch.cat([h["bus"][g.shunt_bus], h["shunt"]], dim=-1))

            h["bus"] = h["bus"] + self.bus_upd[t](torch.cat([h["bus"], bus_agg], dim=-1))
            h["gen"] = h["gen"] + self.gen_upd[t](torch.cat([h["gen"], gen_agg], dim=-1))
            h["load"] = h["load"] + self.load_upd[t](torch.cat([h["load"], load_agg], dim=-1))
            h["shunt"] = h["shunt"] + self.shunt_upd[t](torch.cat([h["shunt"], shunt_agg], dim=-1))
        return h

    def _derive_branch_flows(
        self, g: GraphBatch, bus_vm: torch.Tensor, bus_va: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        Vi = pairwise_complex_voltage(bus_vm, bus_va)

        line_r = g.line_x[:, LINE_R_IDX]
        line_x = g.line_x[:, LINE_X_IDX]
        y_line = complex_admittance_from_impedance(line_r, line_x)
        ycf = torch.complex(torch.zeros_like(g.line_x[:, LINE_BFR_IDX]), g.line_x[:, LINE_BFR_IDX])
        yct = torch.complex(torch.zeros_like(g.line_x[:, LINE_BTO_IDX]), g.line_x[:, LINE_BTO_IDX])
        T_line = torch.ones_like(y_line)

        Vf = Vi[g.line_from]
        Vt = Vi[g.line_to]
        s_f = (y_line + ycf).conj() * (Vf.abs() ** 2) - y_line.conj() * Vf * Vt.conj() / T_line
        s_t = (y_line + yct).conj() * (Vt.abs() ** 2) - y_line.conj() * Vt * Vf.conj() / T_line.conj()

        trafo_r = g.trafo_x[:, TRAFO_R_IDX]
        trafo_x = g.trafo_x[:, TRAFO_X_IDX]
        y_trafo = complex_admittance_from_impedance(trafo_r, trafo_x)
        tycf = torch.complex(torch.zeros_like(g.trafo_x[:, TRAFO_BFR_IDX]), g.trafo_x[:, TRAFO_BFR_IDX])
        tyct = torch.complex(torch.zeros_like(g.trafo_x[:, TRAFO_BTO_IDX]), g.trafo_x[:, TRAFO_BTO_IDX])
        tap = g.trafo_x[:, TRAFO_TAP_IDX].clamp_min(1e-4)
        shift = g.trafo_x[:, TRAFO_SHIFT_IDX]
        T = torch.polar(tap, shift)

        tVf = Vi[g.trafo_from]
        tVt = Vi[g.trafo_to]
        ts_f = (y_trafo + tycf).conj() * (tVf.abs() ** 2) / (tap ** 2) - y_trafo.conj() * tVf * tVt.conj() / T
        ts_t = (y_trafo + tyct).conj() * (tVt.abs() ** 2) - y_trafo.conj() * tVt * tVf.conj() / T.conj()

        return {
            "line_pf": s_f.real,
            "line_qf": s_f.imag,
            "line_pt": s_t.real,
            "line_qt": s_t.imag,
            "trafo_pf": ts_f.real,
            "trafo_qf": ts_f.imag,
            "trafo_pt": ts_t.real,
            "trafo_qt": ts_t.imag,
        }

    def forward(self, g: GraphBatch) -> Dict[str, torch.Tensor]:
        h = {
            "bus": self.enc_bus(g.bus_x),
            "gen": self.enc_gen(g.gen_x),
            "load": self.enc_load(g.load_x),
            "shunt": self.enc_shunt(g.shunt_x),
            "line": self.enc_line(g.line_x),
            "trafo": self.enc_trafo(g.trafo_x),
        }
        h = self._process(g, h)

        bus_raw = self.bus_decoder(h["bus"])
        gen_raw = self.gen_decoder(h["gen"])

        bus_va = bus_raw[:, 0]
        bus_vm = bounded_sigmoid(bus_raw[:, 1], g.bus_vm_min, g.bus_vm_max)
        gen_pg = bounded_sigmoid(gen_raw[:, 0], g.gen_pg_min, g.gen_pg_max)
        gen_qg = bounded_sigmoid(gen_raw[:, 1], g.gen_qg_min, g.gen_qg_max)

        out = {
            "bus_va": bus_va,
            "bus_vm": bus_vm,
            "gen_pg": gen_pg,
            "gen_qg": gen_qg,
        }
        out.update(self._derive_branch_flows(g, bus_vm=bus_vm, bus_va=bus_va))
        return out
