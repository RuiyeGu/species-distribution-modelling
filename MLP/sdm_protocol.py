"""
sdm_protocol.py  (Option B — loads the team's canonical plot-level split)
=========================================================================
SHARED EXPERIMENTAL PROTOCOL for the EcoStat Modelling thesis.

CHANGE FROM EARLIER VERSIONS
----------------------------
This version no longer creates its own train/test split. It LOADS the team's
canonical plot-level split (Ang's), so every model — MLP, RF, XGBoost, LR, GAM —
is evaluated on the identical held-out plots. That is what makes the Direction A
comparison valid.

  * TEST plots come from test_plotlevel.csv  (fixed, shared, never used to tune).
  * DEV  plots come from train_plotlevel.csv (all tuning happens here).
  * CV folds (random AND spatial) are built ON THE DEV PLOTS ONLY, so the test
    set is never touched during cross-validation. Random vs spatial is the
    Direction E comparison; both ride on the same dev plots.

The files are named unambiguously (train_plotlevel / test_plotlevel) on purpose:
two teammates previously wrote different splits to the same `train_split.csv`
name, which is how a row-level (leaky) file ended up in circulation. Load only
the plot-level files here.

The data has 8 species per plot, so the wide pivot is clean (no NaN).
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    log_loss, roc_auc_score, brier_score_loss, f1_score,
    recall_score, confusion_matrix,
)

# ---- canonical split files (place these in the working directory) ----
TRAIN_FILE = "train_plotlevel.csv"
TEST_FILE = "test_plotlevel.csv"

# ---- configuration ----
ENV_FEATURES = ["disturb", "rainann", "soildepth", "soilfert", "tempann", "topo"]
COORD_FEATURES = ["easting", "northing"]
SPECIES = [
    "Eulamprus murrayi", "Saltuarius swaini", "Egernia mcpheei",
    "Pseudechis porphyricaus", "Cacophis kreftii", "Calyptotis scutirostrum",
    "Coeranoscincus reticulatus", "Ophioscincus truncatus",
]
PLOT_COL, TARGET_COL = "plot", "pres.abs"
N_FOLDS = 5
RANDOM_SEED = 42
EPS = 1e-15
SPLIT_FILE = "protocol_splits.json"


def _pivot_wide(df):
    feat = (df.drop_duplicates(PLOT_COL)
              .set_index(PLOT_COL)[ENV_FEATURES + COORD_FEATURES + ["long", "lat"]])
    pa = df.pivot(index=PLOT_COL, columns="Species", values=TARGET_COL)[SPECIES]
    return feat.join(pa).reset_index()


def load_wide(train_path=TRAIN_FILE, test_path=TEST_FILE):
    """Load BOTH canonical files and return one wide table (all plots, one row
    per plot). Membership in dev vs test is decided by the split file, below."""
    df = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)],
                   ignore_index=True)
    return _pivot_wide(df)


def _spatial_blocks(wide_dev, n_per_axis=8):
    e = pd.qcut(wide_dev["easting"], n_per_axis, labels=False, duplicates="drop")
    n = pd.qcut(wide_dev["northing"], n_per_axis, labels=False, duplicates="drop")
    b = e.astype(int) * 1000 + n.astype(int)
    codes = {v: i for i, v in enumerate(sorted(b.unique()))}
    return b.map(codes).values


def make_splits(train_path=TRAIN_FILE, test_path=TEST_FILE, n_per_axis=8):
    """Derive test plots from the test file, and build random + spatial CV folds
    on the training plots only. No splitting happens here — the split is Ang's;
    we only construct the CV folds on top of it."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_plots = sorted(test_df[PLOT_COL].unique().tolist())
    dev_plots = sorted(train_df[PLOT_COL].unique().tolist())

    rng = np.random.RandomState(RANDOM_SEED)
    dev = np.array(dev_plots)

    # random 5-fold on dev plots
    perm = dev.copy(); rng.shuffle(perm)
    random_folds = [perm[i::N_FOLDS].tolist() for i in range(N_FOLDS)]

    # spatial 5-fold on dev plots (group by spatial block)
    wide_dev = _pivot_wide(train_df)
    wide_dev = wide_dev.set_index(PLOT_COL).loc[dev].reset_index()
    blocks = _spatial_blocks(wide_dev, n_per_axis)
    gkf = GroupKFold(n_splits=N_FOLDS)
    spatial_folds = [wide_dev[PLOT_COL].values[val].tolist()
                     for _, val in gkf.split(wide_dev[PLOT_COL].values, groups=blocks)]

    return {
        "description": "Canonical plot-level split (Ang's). Test plots from "
                       "test_plotlevel.csv; CV folds built on train plots only.",
        "random_seed": RANDOM_SEED, "n_folds": N_FOLDS, "n_per_axis": n_per_axis,
        "test_plots": test_plots,
        "random_folds": random_folds,
        "spatial_folds": spatial_folds,
    }


def save_splits(splits, path=SPLIT_FILE):
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved {path}")


def load_splits(path=SPLIT_FILE):
    with open(path) as f:
        return json.load(f)


def iter_cv(wide, splits, scheme="spatial"):
    """Yield (fold, train_df, val_df) for the chosen CV scheme, on dev plots."""
    folds = splits[f"{scheme}_folds"]
    test = set(splits["test_plots"])
    dev = wide[~wide[PLOT_COL].isin(test)]
    for k, val_plots in enumerate(folds):
        vp = set(val_plots)
        yield k, dev[~dev[PLOT_COL].isin(vp)], dev[dev[PLOT_COL].isin(vp)]


def get_dev(wide, splits):
    return wide[~wide[PLOT_COL].isin(set(splits["test_plots"]))]


def get_test(wide, splits):
    return wide[wide[PLOT_COL].isin(set(splits["test_plots"]))]


# ---- shared metric suite ----
def evaluate(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, float), EPS, 1 - EPS)
    y_pred = (y_prob >= threshold).astype(int)
    out = {"n": len(y_true), "n_pos": int(y_true.sum()),
           "prevalence": float(y_true.mean())}
    try:
        out["log_loss"] = log_loss(y_true, y_prob, labels=[0, 1])
    except Exception:
        out["log_loss"] = np.nan
    out["brier"] = brier_score_loss(y_true, y_prob) if len(y_true) else np.nan
    out["auc"] = (roc_auc_score(y_true, y_prob)
                  if y_true.min() != y_true.max() else np.nan)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["sensitivity"] = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    return out


def evaluate_all_species(y_true_df, y_prob_df, threshold=0.5):
    rows = {sp: evaluate(y_true_df[sp].values, y_prob_df[sp].values, threshold)
            for sp in SPECIES}
    res = pd.DataFrame(rows).T
    mc = ["log_loss", "brier", "auc", "f1", "sensitivity", "specificity"]
    res.loc["MACRO_AVG", mc] = res[mc].mean()
    w = res.loc[SPECIES, "prevalence"].values; w = w / w.sum()
    res.loc["PREV_WEIGHTED", mc] = res.loc[SPECIES, mc].multiply(w, axis=0).sum()
    return res


if __name__ == "__main__":
    splits = make_splits()
    save_splits(splits)
    wide = load_wide()
    test = set(splits["test_plots"]); dev = set(wide[PLOT_COL]) - test
    assert test.isdisjoint(dev), "test/dev overlap!"
    print(f"\nPlots: {len(wide)}  dev={len(dev)}  test={len(test)}  overlap=0")
    print("Random fold sizes :", [len(f) for f in splits["random_folds"]])
    print("Spatial fold sizes:", [len(f) for f in splits["spatial_folds"]])
    wtest = get_test(wide, splits)
    print("\nTest-set presence counts per species:")
    for sp in SPECIES:
        print(f"  {sp:30s} {int(wtest[sp].sum()):3d}")
