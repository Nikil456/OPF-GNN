# Project data (Sharaf / case57)

This directory is the **primary dataset location** for this project going forward.

## Layout

- **Sample folders** — One folder per OPF sample, e.g.  
  `pglib_opf_case57_ieee_sample_000009/`  
  Each folder holds CSV exports for that run (e.g. `buses.csv`, `branches.csv`, `generators.csv`, `summary.csv`). More samples can be added as sibling folders here.

- **Index CSV (not a folder)** — The file **`samples_index.csv`** sits **next to** the sample folders (same parent directory, `data_from_sharaf/`). It is only a **catalog** to organize and list metadata across all subfolders (indices, run IDs, objectives, load, line counts, etc.). It does not contain the grid tables themselves.

To find a sample by index, match `sample_index` / `run_id` in `samples_index.csv` to the corresponding `*_sample_NNNNNN` folder name.

## Training with this repo

CSV folders are converted to `GraphBatch` by `canos.sharaf_csv.graphbatch_from_sharaf_dir`. Example:

```bash
source .venv/bin/activate
python scripts/train.py --config configs/canos_sharaf_case57.yaml --device cpu --steps 500
```

Optional: set `dataset.sample_dirs` in the YAML to a list of subfolder names to train on specific samples only.
