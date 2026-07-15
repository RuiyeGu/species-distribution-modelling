# MLP & Validation — EcoStat Modelling (DATA5925)

Owner: Aakash. Covers the per-species MLP, the single-vs-multi-species MLP
comparison, and the shared evaluation protocol (Direction A + Direction E).

All models use the team's **canonical plot-level split** (Ang's), so results are
directly comparable across the whole team. No script re-splits the data.

---

## 1. Setup

Place these files in the working directory:

| File | What it is |
|------|------------|
| `train_plotlevel.csv` | canonical training split (plot-level, 512 plots) |
| `test_plotlevel.csv`  | canonical held-out test split (129 plots) |
| `sdm_protocol.py`     | loads the split, builds CV folds, defines metric suite |
| `mlp_model.py`        | per-species MLP |
| `mlp_single_vs_multi.py` | three-strategy comparison + OOF + Kaggle submissions |
| `class_weight_ablation.py` | class-weighting ablation + calibration figure |
| `kaggle_test.csv` | Kaggle unlabelled test set (2,936 rows) — for submissions |

> **Important — filenames.** Use `train_plotlevel.csv` / `test_plotlevel.csv`,
> NOT `train_split.csv` / `test_split.csv`. Two teammates previously wrote
> different splits (one plot-level, one row-level, which leaks) to the
> `train_split.csv` name. The `_plotlevel` names avoid that collision. The
> canonical split is Ang's plot-level one.

Install dependencies:

```bash
pip3 install numpy pandas scikit-learn matplotlib xgboost
```

(Recommended: a dedicated environment, then `pip3 freeze > requirements.txt`
committed to the repo — results shift between library versions, so pinning is
needed for reproducibility.)

---

## 2. Run order

The scripts have dependencies — run them in this order.

```bash
# STEP 1 — build CV folds from the canonical split (run once).
# Produces protocol_splits.json. Commit that file so everyone's folds match.
python3 sdm_protocol.py

# STEP 2 — per-species MLP.
# Produces mlp_test_results.csv + mlp_selected_configs.csv.
python3 mlp_model.py

# STEP 3 — single vs multi-species (3 strategies) + OOF + Kaggle submissions.
# Independent of step 2. Needs kaggle_test.csv for the submission step.
python3 mlp_single_vs_multi.py

# STEP 4 — class-weighting ablation + calibration figure. Independent.
python3 class_weight_ablation.py
```

`protocol_splits.json` must exist before steps 2–3. Steps 2, 3 and 4 are
otherwise independent of each other.

---

## 3. What each script does

- **`sdm_protocol.py`** — loads the two canonical CSVs, pivots to wide (one row
  per plot, 8 species columns), builds RANDOM and SPATIAL 5-fold CV folds on the
  512 training plots only (test plots never touched), and defines the shared
  metric suite. Run it once; commit the resulting `protocol_splits.json`.

- **`mlp_model.py`** — one MLP per species. Features: 6 environmental +
  easting/northing (`USE_COORDS = True`). No class weighting. Tuned by spatial-CV
  log loss; threshold tuned for F1. Reports log loss, AUC, Brier, F1,
  sensitivity, specificity per species + macro + prevalence-weighted.

- **`mlp_single_vs_multi.py`** — three strategies on one split:
  `single` (per-species), `multi` (shared trunk), `multi_int` (pooled one-hot,
  matching how RF/XGB/LR/GAM do "multi"). Output format mirrors Ang's
  `rf_single_vs_multi_results.csv` for the cross-model table.

---

## 4. Configuration knobs

At the top of `mlp_model.py` and `mlp_single_vs_multi.py`:

- `USE_COORDS` — `True` (spec default, includes easting/northing) or `False`
  (the Direction C no-coordinate experiment). Run both to measure the effect.
- `CV_SCHEME` — `"spatial"` (default) or `"random"`. Switching this is the
  Direction E comparison; the folds for both live in `protocol_splits.json`.
- `GRID` — hyperparameter search space. **Report the full grid in the methods
  section.** If shrunk for a quick run, restore it before reporting anything.

---

## 5. Notes / caveats

- Runtime: the full grid (32 configs × 5 folds × 8 species) takes a few minutes
  per script. Shrink `GRID` while iterating; restore for final runs.
- Convergence warnings from scikit-learn on rare-species folds are expected and
  suppressed; they are not errors.
- Rare species (Eulamprus, Saltuarius, Egernia, Pseudechis — <5 test presences)
  produce noise-dominated per-species metrics. Reported, but not to be
  interpreted individually.
- Single-run margins can shift between seeds and library versions. Confirm any
  reported difference with a multi-seed run before putting it in a table.
