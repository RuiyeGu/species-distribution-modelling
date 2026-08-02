"""
Random Forest baseline for the DATA5925 reptile SDM project.

This script:
  1. Loads the team's canonical plot-level split
     (MLP/train_plotlevel.csv and MLP/test_plotlevel.csv).
  2. Trains one Random Forest classifier per species (single-species approach).
  3. Handles severe class imbalance with class_weight='balanced'.
  4. Evaluates with log loss, AUC-ROC, Brier score, F1, sensitivity, specificity.
  5. Produces a Kaggle-ready submission file.

The canonical split is maintained by the team in the MLP folder; do not
re-split the data here, so that all team models are evaluated identically.

Run from the repository root with the virtual environment activated:
    cd "Random Forest"
    python rf_baseline.py

Author: EcoStat Modelling
"""

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

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # Random Forest/
REPO_ROOT = SCRIPT_DIR.parent                          # repo root
PROJECT_ROOT = REPO_ROOT.parent                        # DATA5905 root
DATA_DIR = PROJECT_ROOT / "predicting-small-reptile-species-distributions-in-nsw"

# Canonical team split (maintained in the MLP folder)
TRAIN_PATH = REPO_ROOT / "MLP" / "train_plotlevel.csv"
TEST_PATH = REPO_ROOT / "MLP" / "test_plotlevel.csv"

KAGGLE_TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = SCRIPT_DIR / "submission_rf_baseline.csv"
SUMMARY_PATH = SCRIPT_DIR / "split_summary.csv"

# ---------------------------------------------------------------------------
# 2. Load the canonical plot-level split
# ---------------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(TEST_PATH)
test = pd.read_csv(KAGGLE_TEST_PATH)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Kaggle test shape:", test.shape)
print("Species:", sorted(train_df["Species"].unique()))
print(
    "Plot-level split -> train plots: {}, val plots: {} (overlap: {})".format(
        train_df["plot"].nunique(),
        val_df["plot"].nunique(),
        len(set(train_df["plot"]) & set(val_df["plot"])),
    )
)

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

species_list = sorted(train_df[SPECIES_COL].unique())

# ---------------------------------------------------------------------------
# 3b. Split summary report (matches team convention)
# ---------------------------------------------------------------------------
full = pd.concat([train_df, val_df], ignore_index=True)
print(f"\nSplit summary (rows per species and class):")
summary_rows = []
for sp in species_list:
    for cls, label in [(0, "negative"), (1, "positive")]:
        total = int(((full[SPECIES_COL] == sp) & (full[TARGET_COL] == cls)).sum())
        tr_n = int(((train_df[SPECIES_COL] == sp) & (train_df[TARGET_COL] == cls)).sum())
        va_n = int(((val_df[SPECIES_COL] == sp) & (val_df[TARGET_COL] == cls)).sum())
        summary_rows.append(
            {"Species": sp, "class": label, "total": total, "train": tr_n, "test": va_n}
        )
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(SUMMARY_PATH, index=False)
print(f"Saved {SUMMARY_PATH.name}")

print(
    f"\nTotal -> train: {len(train_df)} rows (presence rate {train_df[TARGET_COL].mean():.4f}), "
    f"test: {len(val_df)} rows (presence rate {val_df[TARGET_COL].mean():.4f})"
)

# ---------------------------------------------------------------------------
# 4. Train one Random Forest per species
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
# 5. Aggregate validation performance
# ---------------------------------------------------------------------------
val_all = pd.concat(val_predictions, ignore_index=True)
y_true_all = val_all[TARGET_COL]
y_prob_all = val_all["pred_prob"]

overall_log_loss = log_loss(y_true_all, y_prob_all, labels=[0, 1])
overall_auc = roc_auc_score(y_true_all, y_prob_all)
overall_brier = brier_score_loss(y_true_all, y_prob_all)

print("\n" + "=" * 60)
print("Overall validation performance (canonical plot-level split)")
print("=" * 60)
print(f"Log loss      : {overall_log_loss:.5f}")
print(f"AUC-ROC       : {overall_auc:.4f}")
print(f"Brier score   : {overall_brier:.4f}")

results_df = pd.DataFrame(species_results)
print("\nPer-species summary:")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Kaggle submission
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
