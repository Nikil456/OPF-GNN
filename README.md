# CANOS Reproduction (Paper Implementation)

This repository contains a practical reproduction of the core CANOS method from:

- *CANOS: A Fast and Scalable Neural AC-OPF Solver Robust To N-1 Perturbations* (arXiv:2403.17660)

It implements the key architectural and training ideas from the paper:

- Encode-Process-Decode GNN for heterogeneous power-grid graphs
- Typed message passing with many interaction steps and residual connections
- Bounded outputs for voltage magnitude and generator dispatch via sigmoid mapping
- Branch-flow derivation from predicted voltages using AC branch equations
- Constraint-augmented training objective:
  - supervised L2 + weighted constraint violations

It can now also consume the real PyTorch Geometric OPF dataset via
`torch_geometric.datasets.OPFDataset`.

## What this reproduction includes

- `src/canos/model.py`: CANOS architecture
- `src/canos/losses.py`: supervised and constraint losses
- `src/canos/data.py`: graph batch schema and synthetic dataset for smoke runs
- `scripts/train.py`: training script
- `configs/canos_small.yaml`: default config

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you are using the PyG OPF dataset path, `torch-geometric` is now included in
the project dependencies.

## Run

```bash
python scripts/train.py --config configs/canos_small.yaml --device cpu
```

A checkpoint is written to `artifacts/canos_repro.pt`.

## Run With PyG OPFDataset

Install PyTorch Geometric in the active environment, then run:

```bash
python scripts/train.py --config configs/canos_pyg_opf.yaml --device cpu
```

This uses `torch_geometric.datasets.OPFDataset` and maps the returned
`HeteroData` into the internal `GraphBatch` format expected by CANOS.

The training path also applies a compatibility patch for Python 3.10/3.11,
since current PyG OPFDataset releases may call tar extraction with a keyword
that is only supported in newer Python versions.

Supported OPFDataset fields used by the adapter:

- Node types: `bus`, `generator`, `load`, `shunt`
- Edge types: `("bus", "ac_line", "bus")`, `("bus", "transformer", "bus")`
- Link edges: `("generator", "generator_link", "bus")`, `("load", "load_link", "bus")`, `("shunt", "shunt_link", "bus")`
- Targets: node `y` for bus/generator and `edge_label` for ac lines/transformers

## Mapping to paper components

1. **Encode**
   - Independent linear projections per node/edge type into latent vectors.
2. **Process**
   - Multiple typed interaction steps over bus/gen/load/shunt/line/transformer entities.
3. **Decode**
   - Bus decoder predicts `va`, `vm`; generator decoder predicts `pg`, `qg`.
4. **Bounds enforcement**
   - `sigmoid(raw) * (upper - lower) + lower` on `vm`, `pg`, `qg`.
5. **Derive branch flow**
   - Uses branch electrical parameters and predicted voltages for `pf, qf, pt, qt`.
6. **Training objective**
   - `L = L_supervised + C * L_constraints`, with `C=0.1` default.

## Notes on fidelity

- This is a clean reproduction-oriented implementation in PyTorch.
- The paper's exact software stack used JAX/Haiku/Jraph; this code mirrors method logic rather than exact framework internals.
- The synthetic dataset is for smoke-testing only. Replace it with real PGLIB-derived graph tensors and AC-OPF labels to replicate reported metrics.
- For PyG OPFDataset, feature dimensions are inferred from the dataset sample at runtime, so the model matches the dataset schema automatically.
- The train script runs exactly one dataset configuration at a time: one split, one case name, one perturbation setting. It does not iterate over all PGLIB cases automatically.