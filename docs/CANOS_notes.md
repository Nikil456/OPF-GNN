# CANOS in simple terms: this repo, training, Sharaf data, and what we cannot do yet

This note is for our team. It explains [CANOS](https://arxiv.org/abs/2403.17660), how we wired training here, what Sharaf CSV does, and **why line-switching “probability” is not something the current code learns**.

---

## 1. What CANOS is (short)

- The grid is a **graph** (buses, lines, generators, …).
- A GNN runs **encode → message passing → decode** and predicts an **AC-OPF–style** answer: voltages, generation, and (via physics) branch flows.
- Flows on lines are often **computed from voltages** and line data, not guessed by a big MLP at the end, so the result stays closer to real physics.

---

## 2. How training runs in this repo

1. **Config** — YAML in [`configs/`](../configs/) (e.g. [`canos_sharaf_case57.yaml`](../configs/canos_sharaf_case57.yaml)).
2. **Dataset** — **`sharaf_csv` only**: folders under [`data_from_sharaf/`](../data_from_sharaf/README.md) → **`GraphBatch`** via `graphbatch_from_sharaf_dir()` in [`src/canos/sharaf_csv.py`](../src/canos/sharaf_csv.py) ([`GraphBatch`](../src/canos/data.py) schema).
3. **Model** — [`CANOS`](../src/canos/model.py): message passing, then **Vm, Pg, Qg**, then **line/trafo flows** from physics.
4. **Loss** — [`compute_total_loss`](../src/canos/losses.py): **MSE on targets** + **0.1 × constraint violations** (same idea as the paper’s **C = 0.1**).
5. **Checkpoint** — saved by [`scripts/train.py`](../scripts/train.py).

Quick check without long training: [`scripts/verify_canos_outputs.py`](../scripts/verify_canos_outputs.py).

### Training is working: what we see on the terminal

The train loop runs end-to-end: forward pass, **`compute_total_loss`**, backward, optimizer step, then periodic logging and a final **`torch.save`**. When things are healthy, **`total`** and **`sup`** (supervised MSE) trend **down** over steps; **`cons`** (constraint penalty) can move but should stay **finite** (no NaNs from empty tensors after our `losses.py` fix).

Example console output — **line format is fixed** in [`scripts/train.py`](../scripts/train.py); **numeric values** depend on data, device, and step count. With [`canos_sharaf_case57.yaml`](../configs/canos_sharaf_case57.yaml), `log_every: 25` so we see a line every 25 steps (and always on step 1). A typical healthy run shows **`total`** and **`sup`** decreasing:

```text
step=00001 total=18.432901 sup=2.184301 cons=16.248600 time=0.1s
step=00025 total=5.210443 sup=1.923401 cons=3.287042 time=2.3s
step=00050 total=2.891204 sup=1.102884 cons=1.788320 time=4.6s
step=00075 total=1.654302 sup=0.712104 cons=0.942198 time=6.9s
step=00100 total=1.021887 sup=0.501203 cons=0.520684 time=9.2s
...
saved checkpoint: artifacts/sharaf_case57/canos_repro.pt
```

Reproduce (from repo root; optional **`--steps`** overrides YAML for a quick smoke test):

```bash
python scripts/train.py --config configs/canos_sharaf_case57.yaml --device cpu
# python scripts/train.py --config configs/canos_sharaf_case57.yaml --device cpu --steps 100
```

---

## 3. Sharaf CSV: what we did

We added **`graphbatch_from_sharaf_dir()`** so one sample folder (`buses.csv`, `branches.csv`, `generators.csv`) becomes a **`GraphBatch`**:

- Convert **MW → per unit**, **degrees → radians** where needed.
- Skip lines with **`line_enabled == 0`**.
- If there are **no transformers** in the file, trafo tensors are **empty** (length 0).
- **Targets** for training come from **`label_*`** columns (voltages, powers, flows), not from switching.

We also fixed **`constraint_loss`**: **`mean()` on empty tensors** was **NaN**; we use **`_mean_finite()`** in [`losses.py`](../src/canos/losses.py).

**`samples_index.csv`** only lists samples; it does **not** replace reading each folder’s CSVs.

YAML hook: `dataset.type: sharaf_csv` — see [`scripts/train.py`](../scripts/train.py) (around the `sharaf_csv` branch) and [`canos_sharaf_case57.yaml`](../configs/canos_sharaf_case57.yaml).

---

## 4. What the model outputs today (no classification)

The model only outputs **numbers** (regression), for example:

- **`bus_va`, `bus_vm`** — per bus  
- **`gen_pg`, `gen_qg`** — per generator  
- **`line_pf` … `line_qt`** — per line  
- **`trafo_*`** — per trafo edge (or empty)

There is **no** output like “probability this line is open” and **no** classification layer for lines.

---

## 5. Line switching / “trip probability”: labels, data limits, and a future path (read this)

Everything below is **one story**: why we cannot learn “which line is off” from the current model and exports, what would have to change, and a sketch of future work. **Not coded yet** except where the repo already does regression.

### Why this section exists

Questions like **“which line is off”** or **“trip probability”** need **supervised classification** (or a probabilistic variant). That is **not** what the current **`CANOS`** head does; it outputs **continuous** state only (see §4).

### What supervised classification needs

- For **every training example** (every scenario / sample), we need a **fixed list of lines** (branches) and **one label per line**.
- The usual form is **0 or 1 per line** (example: **1 = line in service**, **0 = line out**). We pick the rule once and stick to it.
- That **per-line vector** is the **ground truth**. Without it, there is **no proper classification target**.

### What Sharaf-style exports give us today

- **`samples_index.csv`** often has **`lines_on` and `lines_off` as two numbers** — **counts** only (“how many on/off”), **not** “which” branches.
- **`line_enabled` in `branches.csv`** tells us the graph **for that one export**, not “in scenario *k*, line *e* was tripped” across many runs.

**So:** we **do not** have a **per-line 0/1 vector per sample** in the pipeline described in §3. Counts like `lines_on` / `lines_off` are **not** that answer key.

### Sample count vs missing labels

- **Few sample folders** → less data for **regression** (we may only train on one scenario until more folders are added).
- **Many** folders still **do not** fix classification: **more OPF samples** help **regression** (voltages, flows across loads), but **they do not replace** missing **per-line switching labels**. If every sample still has **no** branch-wise on/off vector, we still **cannot** train a line-switching classifier **from labels alone**.

**Bottom line for “why isn’t classification here?”** Not only because we might have “one sample”, but because **(1) the model has no classification head**, and **(2) the exports, as we use them, lack the per-line answer key classification needs.**

### If Sharaf adds more data later

- **More folders with OPF labels** → better **continuous** predictions (voltages, flows) — **same pipeline**, more files.
- **Switching / probability** → we need **new columns or files** (per sample: **branch-wise 0/1 or probability**), plus **new model code** (classifier head, BCE, etc.). **Not** in this repo yet.

### Future classification (plan only)

Nothing here is wired for import; it is a **roadmap**.

1. **Decide** what “probability” means (on vs trip vs uncertainty).
2. **Collect** per-line labels per scenario (simulation, SCOPF, N-1 tables, history — **not** from count columns alone).
3. **Add** a small **head**: one logit per line → sigmoid.
4. **Train** with **BCE** (or similar). Optionally **two stages**: pick topology first, then run CANOS on that graph only.

---

## 6. Paper loss in plain symbols

```text
loss = loss_supervised + C * loss_constraints
```

**C** is **0.1** in the paper; we use **`constraint_weight: 0.1`** in `compute_total_loss`.

---

## 7. Where to look in the code

| What | File |
|------|------|
| Model | [`src/canos/model.py`](../src/canos/model.py) |
| `GraphBatch` | [`src/canos/data.py`](../src/canos/data.py) |
| Sharaf → `GraphBatch` | [`src/canos/sharaf_csv.py`](../src/canos/sharaf_csv.py) |
| Losses | [`src/canos/losses.py`](../src/canos/losses.py) |
| Train script | [`scripts/train.py`](../scripts/train.py) |
| Data folder layout | [`data_from_sharaf/README.md`](../data_from_sharaf/README.md) |

---

## 8. Bottom line

We use **CANOS-style regression** (continuous state + constraint penalty). **Sharaf CSV** is wired for that. **Line on/off or trip probability** needs **different outputs and per-line labels**; **`lines_on` / `lines_off` counts are not enough**, and the **current model does not output class probabilities at all.**
