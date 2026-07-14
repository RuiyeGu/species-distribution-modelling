"""
Random Forest baseline for the DATA5925 reptile SDM project.

This script:
  1. Loads train.csv and test.csv.
  2. Uses plot-level grouped train/validation split to avoid spatial leakage.
  3. Trains one Random Forest classifier per species (single-species approach).
  4. Handles severe class imbalance with class_weight='balanced'.
  5. Evaluates with log loss, AUC-ROC, Brier score, F1, sensitivity, specificity.
  6. Produces a Kaggle-ready submission file.

Run from the project root with the virtual environment activated:
    source .venv_pydata/bin/activate
    python src/rf_baseline.py

Author: EcoStat Modelling
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    brier_score_loss,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = ROOT / "predicting-small-reptile-species-distributions-in-nsw"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = OUT_DIR / "submission_rf_baseline.csv"

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Species:", train["Species"].unique())
print("Overall presence rate: {:.4f}".format(train["pres.abs"].mean()))

# ---------------------------------------------------------------------------
# 3. Feature selection
# ---------------------------------------------------------------------------
# IMPORTANT: use easting/northing (projected Euclidean coordinates) and avoid
# putting both long/lat and easting/northing in the model together.
# disturb is ordinal in the data (1-3); we keep it numeric for the baseline.
FEATURE_COLS = [
    "easting",
    "northing",
    "disturb",
    "rainann",
    "soildepth",
    "soilfert",
    "tempann",
    "topo",
]
TARGET_COL = "pres.abs"
GROUP_COL = "plot"
SPECIES_COL = "Species"

# ---------------------------------------------------------------------------
# 4. Plot-level train / validation split
# ---------------------------------------------------------------------------
# Because the same plot contains 8 species rows, splitting by row would leak
# spatial information. We split by plot (GROUP_COL).
species_list = sorted(train[SPECIES_COL].unique())

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(train, groups=train[GROUP_COL]))

train_df = train.iloc[train_idx].copy()
val_df = train.iloc[val_idx].copy()

print(
    "Plot-level split -> train plots: {}, val plots: {}".format(
        train_df[GROUP_COL].nunique(), val_df[GROUP_COL].nunique()
    )
)

# ---------------------------------------------------------------------------
# 4b. Split summary report (matches team convention)
# ---------------------------------------------------------------------------
print(f"\nSplit summary (rows per species and class):")
summary_rows = []
for sp in species_list:
    for cls, label in [(0, "negative"), (1, "positive")]:
        total = int(((train[SPECIES_COL] == sp) & (train[TARGET_COL] == cls)).sum())
        tr_n = int(((train_df[SPECIES_COL] == sp) & (train_df[TARGET_COL] == cls)).sum())
        va_n = int(((val_df[SPECIES_COL] == sp) & (val_df[TARGET_COL] == cls)).sum())
        summary_rows.append(
            {"Species": sp, "class": label, "total": total, "train": tr_n, "test": va_n}
        )
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# Save the split summary table for reporting / sharing with the team
summary_df.to_csv(OUT_DIR / "split_summary.csv", index=False)
print("Saved split_summary.csv")

tr_pres = train_df[TARGET_COL].mean()
va_pres = val_df[TARGET_COL].mean()
print(
    f"\nTotal -> train: {len(train_df)} rows (presence rate {tr_pres:.4f}), "
    f"test: {len(val_df)} rows (presence rate {va_pres:.4f})"
)

# Save the split for reproducibility and team-wide consistency
train_df.to_csv(OUT_DIR / "train_split.csv", index=False)
val_df.to_csv(OUT_DIR / "test_split.csv", index=False)
print("\nSaved train_split.csv and test_split.csv")

# ---------------------------------------------------------------------------
# 5. Train one Random Forest per species
# ---------------------------------------------------------------------------
val_predictions = []
species_results = []
models = {}

for sp in species_list:
    tr_sp = train_df[train_df[SPECIES_COL] == sp]
    va_sp = val_df[val_df[SPECIES_COL] == sp]

    X_tr = tr_sp[FEATURE_COLS]
    y_tr = tr_sp[TARGET_COL]
    X_va = va_sp[FEATURE_COLS]
    y_va = va_sp[TARGET_COL]

    # class_weight='balanced' automatically adjusts for the rare-species issue.
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    models[sp] = clf

    # Predicted probability of presence (class 1), clipped to avoid log(0)
    prob = np.clip(clf.predict_proba(X_va)[:, 1], 1e-6, 1 - 1e-6)
    pred_label = clf.predict(X_va)

    va_sp = va_sp.copy()
    va_sp["pred_prob"] = prob
    val_predictions.append(va_sp)

    # Metrics
    auc = roc_auc_score(y_va, prob) if len(np.unique(y_va)) > 1 else np.nan
    brier = brier_score_loss(y_va, prob)
    f1 = f1_score(y_va, pred_label, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_va, pred_label).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    logloss = log_loss(y_va, prob, labels=[0, 1])

    species_results.append(
        {
            "species": sp,
            "n_train": len(y_tr),
            "n_val": len(y_va),
            "presence_rate_val": y_va.mean(),
            "log_loss": logloss,
            "auc_roc": auc,
            "brier": brier,
            "f1": f1,
            "sensitivity": sens,
            "specificity": spec,
        }
    )

    print(
        f"{sp:30s} | val n={len(y_va):3d} | pres={y_va.mean():.3f} | "
        f"logloss={logloss:.4f} | AUC={auc:.3f} | F1={f1:.3f}"
    )

# ---------------------------------------------------------------------------
# 6. Aggregate validation performance
# ---------------------------------------------------------------------------
val_all = pd.concat(val_predictions, ignore_index=True)
y_true_all = val_all[TARGET_COL]
y_prob_all = val_all["pred_prob"]

overall_log_loss = log_loss(y_true_all, y_prob_all, labels=[0, 1])
overall_auc = roc_auc_score(y_true_all, y_prob_all)
overall_brier = brier_score_loss(y_true_all, y_prob_all)

print("\n" + "=" * 60)
print("Overall validation performance (plot-level split)")
print("=" * 60)
print(f"Log loss      : {overall_log_loss:.5f}")
print(f"AUC-ROC       : {overall_auc:.4f}")
print(f"Brier score   : {overall_brier:.4f}")

results_df = pd.DataFrame(species_results)
print("\nPer-species summary:")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Kaggle submission
# ---------------------------------------------------------------------------
submission_probs = []
for sp in species_list:
    test_sp = test[test[SPECIES_COL] == sp].copy()
    X_te = test_sp[FEATURE_COLS]
    prob = np.clip(models[sp].predict_proba(X_te)[:, 1], 1e-6, 1 - 1e-6)
    test_sp["pred"] = prob
    submission_probs.append(test_sp[["id", "pred"]])

submission = pd.concat(submission_probs, ignore_index=True)
submission = submission.sort_values("id").reset_index(drop=True)
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"\nSubmission saved to: {SUBMISSION_PATH}")
print(submission.head(10))
