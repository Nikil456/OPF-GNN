# OPF-GNN

Deep Learning for AC Optimal Power Flow using PyTorch Geometric (PyG).

## Goal

Replicate and extend the CANOS-style architecture to solve AC-OPF: predict near-optimal solutions (voltage magnitude/angle, generator setpoints) in real time (<100 ms) with GNNs. Target case: **pglib_opf_case14_ieee** (14-bus).

## Data (OPFDataset)

- **torch_geometric.datasets.OPFDataset** — heterogeneous graph:
  - **Nodes:** `bus` (core; x = demand/bounds, y = solution), `generator` (x = cost/limits, y = Pg, Qg), `load`, `shunt`
  - **Edges:** `('bus','ac_line','bus')`, `('bus','transformer','bus')`, and link edges: `generator_link`, `load_link`, `shunt_link` (each in both directions)

## Project layout

- **train.py** — load case14, print summary, train the model (entry point)
- **model.py** — `HeteroGNN` (SAGEConv per edge type, type-specific MLP heads for bus & generator)
- **losses.py** — MSE + placeholder for physics-informed loss (e.g. \(S = V \cdot I^*\))

## Quick start (venv recommended)

From the project root, run these in order. **Step 1 creates the venv (do it once);** step 2 activates it so the shell uses the venv’s Python and pip.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

If you see `no such file or directory: .venv/bin/activate`, run `python3 -m venv .venv` first, then `source .venv/bin/activate`.

Data is downloaded to `data/OPF/` on first run (case14, `num_groups=1`).

## Training

By default, `python train.py` trains for 10 epochs on the train split. Optional arguments:

```bash
python train.py --epochs 20 --lr 5e-4 --hidden 64 --layers 2 --num_groups 1
python train.py --no_summary   # skip data summary, only train
```

Uses CPU or GPU automatically if available.

## Testing

Smoke tests use synthetic graphs (no dataset download). From the project root:

```bash
source .venv/bin/activate   # if using venv
pytest tests/ -v
```

Or run the test file directly:

```bash
python -m pytest tests/test_pipeline.py -v
```

Tests cover: model forward (output shapes), `mse_loss` / `combined_loss`.

## Apple Silicon (M1/M2/M3): PyTorch architecture error

The error `have 'x86_64', need 'arm64'` means PyTorch was installed for x86_64. Reinstall for arm64:

```bash
pip uninstall torch torch-geometric
pip install torch torch-geometric numpy
```

If it persists, check that you are using native arm64 Python:

```bash
python3 -c "import platform; print(platform.machine())"
```

You want `arm64`. If you see `x86_64`, the terminal may be running under Rosetta or using an x86-only Python. Then:

- Run the terminal as a native Apple Silicon app, or  
- Install Python from [python.org](https://www.python.org/downloads/) using the **Apple Silicon** macOS installer and install `torch` with that Python’s `pip`.
