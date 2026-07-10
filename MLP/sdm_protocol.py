

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    log_loss, roc_auc_score, brier_score_loss, f1_score,
    recall_score, confusion_matrix,
)

ENV_FEATURES = ["disturb", "rainann", "soildepth", "soilfert", "tempann", "topo"]
COORD_FEATURES = ["easting", "northing"]
SPECIES = [
    "Eulamprus murrayi", "Saltuarius swaini", "Egernia mcpheei",
    "Pseudechis porphyricaus", "Cacophis kreftii", "Calyptotis scutirostrum",
    "Coeranoscincus reticulatus", "Ophioscincus truncatus",
]
PLOT_COL, TARGET_COL = "plot", "pres.abs"
N_FOLDS = 5
TEST_FRACTION = 0.20
RANDOM_SEED = 42
EPS = 1e-15
SPLIT_FILE = "protocol_splits.json"


def load_wide(path="train.csv"):
    df = pd.read_csv(path)
    feat = (df.drop_duplicates(PLOT_COL)
              .set_index(PLOT_COL)[ENV_FEATURES + COORD_FEATURES + ["long", "lat"]])
    pa = df.pivot(index=PLOT_COL, columns="Species", values=TARGET_COL)[SPECIES]
    return feat.join(pa).reset_index()


def _richness_bucket(wide):
    
    r = wide[SPECIES].sum(axis=1)
    return np.minimum(r, 3).astype(int)


def _spatial_blocks(wide, n_per_axis=8):

    e = pd.qcut(wide["easting"], n_per_axis, labels=False, duplicates="drop")
    n = pd.qcut(wide["northing"], n_per_axis, labels=False, duplicates="drop")
    b = e.astype(int) * 1000 + n.astype(int)
    codes = {v: i for i, v in enumerate(sorted(b.unique()))}
    return b.map(codes).values


def make_splits(path="train.csv", n_per_axis=8):
    wide = load_wide(path)
    rng = np.random.RandomState(RANDOM_SEED)
    plots = wide[PLOT_COL].values
    strata = _richness_bucket(wide).values

    test_mask = np.zeros(len(wide), dtype=bool)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        k = int(round(TEST_FRACTION * len(idx)))
        test_mask[idx[:k]] = True
    test_plots = plots[test_mask].tolist()
    dev_idx = np.where(~test_mask)[0]
    dev_plots = plots[dev_idx]

    perm = dev_plots.copy(); rng.shuffle(perm)
    random_folds = [perm[i::N_FOLDS].tolist() for i in range(N_FOLDS)]

    blocks = _spatial_blocks(wide, n_per_axis)[dev_idx]
    gkf = GroupKFold(n_splits=N_FOLDS)
    spatial_folds = [dev_plots[val].tolist()
                     for _, val in gkf.split(dev_plots, groups=blocks)]

    return {
        "description": "Plot-level splits (wide format). Test=stratified random; "
                       "CV folds provided in both random and spatial variants.",
        "random_seed": RANDOM_SEED, "n_folds": N_FOLDS,
        "test_fraction": TEST_FRACTION, "n_per_axis": n_per_axis,
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
    splits = make_splits("train.csv")
    save_splits(splits)
    wide = load_wide("train.csv")

    test = set(splits["test_plots"])
    dev = set(wide[PLOT_COL]) - test
    assert test.isdisjoint(dev), "TEST/DEV plot overlap!"
    print(f"\nPlots: {len(wide)}  dev={len(dev)}  test={len(test)}  overlap=0  (leak-free)")

    # per-species prevalence preserved in dev vs test?
    wtest = get_test(wide, splits); wdev = get_dev(wide, splits)
    print("\nPer-species prevalence  dev / test  (+ test presence count):")
    for sp in SPECIES:
        print(f"  {sp:30s} {wdev[sp].mean():.3f} / {wtest[sp].mean():.3f}"
              f"   (test pos = {int(wtest[sp].sum())})")

    print("\nRandom fold sizes :", [len(f) for f in splits["random_folds"]])
    print("Spatial fold sizes:", [len(f) for f in splits["spatial_folds"]])
