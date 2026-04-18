# OPF-GNN: CANOS + Sharaf CSV

Reproduction-oriented **CANOS** ([arXiv:2403.17660](https://arxiv.org/abs/2403.17660)) training on **Sharaf-exported CSV** samples: one subfolder per scenario under [`data_from_sharaf/`](data_from_sharaf/README.md) (`buses.csv`, `branches.csv`, `generators.csv`, …).

Concept note for the team: [`docs/CANOS_notes.md`](docs/CANOS_notes.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## How to test

From the repo root, with the venv activated and dependencies installed (see **Setup**).

### 1. Unit-style smoke test (forward, loss, backward)

**Requires** at least one sample folder under `data_from_sharaf/` with `buses.csv` (see [`data_from_sharaf/README.md`](data_from_sharaf/README.md)).

```bash
python3 scripts/verify_canos_outputs.py
```

**Expected:** a single line ending with something like:

```text
verify_canos_outputs: OK (forward, loss, backward)
```

If the directory is missing or empty, the script exits with an error telling you to add CSV samples.

### 2. Training loop smoke test (logged losses + checkpoint)

Runs a few optimizer steps and prints `total`, `sup` (supervised), and `cons` (constraints). Use a small step count so it finishes in seconds:

```bash
python3 scripts/train.py --config configs/canos_sharaf_case57.yaml --device cpu --steps 35
```

**Expected:** lines like:

```text
step=00001 total=... sup=... cons=... time=...s
step=00025 total=... sup=... cons=... time=...s
saved checkpoint: artifacts/sharaf_case57/canos_repro.pt
```

Numeric values depend on data and hardware; the important part is **finite** losses and **no traceback**.

### 3. Full training run

Uses the step count from the YAML (`train.steps`, e.g. 500 in `canos_sharaf_case57.yaml`):

```bash
python3 scripts/train.py --config configs/canos_sharaf_case57.yaml --device cpu
```

Override steps for a longer or shorter run: `--steps N`.

Checkpoint path: `artifacts/sharaf_case57/canos_repro.pt` unless you change `train.out_dir` in the config.

## Implementation map

| Piece | Location |
|--------|----------|
| Model (`CANOS`) | `src/canos/model.py` |
| `GraphBatch` | `src/canos/data.py` |
| Sharaf CSV → `GraphBatch` | `src/canos/sharaf_csv.py` |
| Losses | `src/canos/losses.py` |
| Training loop | `scripts/train.py` |
| Config | `configs/canos_sharaf_case57.yaml` |

## What matches the paper (high level)

- Heterogeneous graph (bus, generator, load, shunt, ac_line, transformer)
- Encode–process–decode GNN, bounded `vm` / `pg` / `qg`, branch flows from physics
- `total = supervised + constraint_weight × constraints` (default **C = 0.1**)

Stack here is **PyTorch** only (no PyG / OPFDataset in this trimmed repo).
