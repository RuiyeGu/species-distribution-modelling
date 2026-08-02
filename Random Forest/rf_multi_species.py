"""
Multi-species Random Forest for the DATA5925 reptile SDM project.

This script compares three Random Forest strategies on the team's canonical
plot-level split (MLP/train_plotlevel.csv and MLP/test_plotlevel.csv):
  1. Single-species: one RF per species (baseline).
  2. Multi-species (one-hot): one RF trained on all species with Species one-hot encoded.
  3. Multi-species (interactions): one-hot Species plus Species x feature interactions.

All approaches are evaluated on the same held-out test set for fair comparison.

The canonical split is maintained by the team in the MLP folder; do not
re-split the data here, so that all team models are evaluated identically.

Run from the repository root with the virtual environment activated:
    cd "Random Forest"
    python rf_multi_species.py

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
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
# This script lives in species-distribution-modelling/Random Forest/.
# The Kaggle data lives in the parent DATA5905 folder.
SCRIPT_DIR = Path(__file__).resolve().parent          # Random Forest/
REPO_ROOT = SCRIPT_DIR.parent                          # repo root
PROJECT_ROOT = REPO_ROOT.parent                        # DATA5905 root
DATA_DIR = PROJECT_ROOT / "predicting-small-reptile-species-distributions-in-nsw"

# Canonical team split (maintained in the MLP folder)
TRAIN_PATH = REPO_ROOT / "MLP" / "train_plotlevel.csv"
TEST_PATH = REPO_ROOT / "MLP" / "test_plotlevel.csv"

KAGGLE_TEST_PATH = DATA_DIR / "test.csv"
RESULTS_PATH = SCRIPT_DIR / "rf_single_vs_multi_results.csv"
SUBMISSION_SINGLE_PATH = SCRIPT_DIR / "submission_rf_single_species.csv"
SUBMISSION_MULTI_PATH = SCRIPT_DIR / "submission_rf_multi_species.csv"

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
print("Overall presence rate: {:.4f}".format(train_df["pres.abs"].mean()))
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
# 4. Feature builders
# ---------------------------------------------------------------------------
def build_single_species_X(df):
    """Numeric features only (used in per-species models)."""
    return df[FEATURE_COLS].copy()


def build_multi_species_X(df, interactions=False):
    """Numeric features + one-hot Species (+ optional interactions)."""
    X = df[FEATURE_COLS].copy().reset_index(drop=True)
    species_dummies = pd.get_dummies(df[SPECIES_COL], prefix="sp").reset_index(drop=True)

    if interactions:
        interaction_features = pd.DataFrame(index=X.index)
        for feat in FEATURE_COLS:
            for sp_col in species_dummies.columns:
                interaction_features[f"{feat}_x_{sp_col}"] = X[feat].values * species_dummies[sp_col].values
        X = pd.concat([X, species_dummies, interaction_features], axis=1)
    else:
        X = pd.concat([X, species_dummies], axis=1)

    return X


def get_common_rf():
    """Common Random Forest configuration for fair comparison."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# 5. Strategy 1: Single-species Random Forest
# ---------------------------------------------------------------------------
single_models = {}
single_val_probs = np.zeros(len(val_df))

print("\n" + "=" * 60)
print("Strategy 1: Single-species Random Forest")
print("=" * 60)

for sp in species_list:
    tr_sp = train_df[train_df[SPECIES_COL] == sp]
    va_mask = val_df[SPECIES_COL] == sp
    va_sp = val_df[va_mask]

    X_tr = build_single_species_X(tr_sp)
    y_tr = tr_sp[TARGET_COL]
    X_va = build_single_species_X(va_sp)

    clf = get_common_rf()
    clf.fit(X_tr, y_tr)
    single_models[sp] = clf

    prob = np.clip(clf.predict_proba(X_va)[:, 1], 1e-6, 1 - 1e-6)
    single_val_probs[va_mask.values] = prob

single_log_loss = log_loss(val_df[TARGET_COL], single_val_probs, labels=[0, 1])
single_auc = roc_auc_score(val_df[TARGET_COL], single_val_probs)
single_brier = brier_score_loss(val_df[TARGET_COL], single_val_probs)

print(f"Overall log loss: {single_log_loss:.5f}")
print(f"Overall AUC-ROC : {single_auc:.4f}")
print(f"Overall Brier   : {single_brier:.4f}")

# ---------------------------------------------------------------------------
# 6. Strategy 2: Multi-species Random Forest (Species one-hot)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Strategy 2: Multi-species Random Forest (Species one-hot)")
print("=" * 60)

X_tr_multi = build_multi_species_X(train_df, interactions=False)
y_tr_multi = train_df[TARGET_COL].reset_index(drop=True)
X_va_multi = build_multi_species_X(val_df, interactions=False)

multi_model = get_common_rf()
multi_model.fit(X_tr_multi, y_tr_multi)

multi_val_probs = np.clip(multi_model.predict_proba(X_va_multi)[:, 1], 1e-6, 1 - 1e-6)
multi_log_loss = log_loss(val_df[TARGET_COL].reset_index(drop=True), multi_val_probs, labels=[0, 1])
multi_auc = roc_auc_score(val_df[TARGET_COL].reset_index(drop=True), multi_val_probs)
multi_brier = brier_score_loss(val_df[TARGET_COL].reset_index(drop=True), multi_val_probs)

print(f"Overall log loss: {multi_log_loss:.5f}")
print(f"Overall AUC-ROC : {multi_auc:.4f}")
print(f"Overall Brier   : {multi_brier:.4f}")

# ---------------------------------------------------------------------------
# 7. Strategy 3: Multi-species Random Forest with interactions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Strategy 3: Multi-species Random Forest (Species one-hot + interactions)")
print("=" * 60)

X_tr_int = build_multi_species_X(train_df, interactions=True)
y_tr_int = train_df[TARGET_COL].reset_index(drop=True)
X_va_int = build_multi_species_X(val_df, interactions=True)

int_model = get_common_rf()
int_model.fit(X_tr_int, y_tr_int)

int_val_probs = np.clip(int_model.predict_proba(X_va_int)[:, 1], 1e-6, 1 - 1e-6)
int_log_loss = log_loss(val_df[TARGET_COL].reset_index(drop=True), int_val_probs, labels=[0, 1])
int_auc = roc_auc_score(val_df[TARGET_COL].reset_index(drop=True), int_val_probs)
int_brier = brier_score_loss(val_df[TARGET_COL].reset_index(drop=True), int_val_probs)

print(f"Overall log loss: {int_log_loss:.5f}")
print(f"Overall AUC-ROC : {int_auc:.4f}")
print(f"Overall Brier   : {int_brier:.4f}")

# ---------------------------------------------------------------------------
# 8. Per-species comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Per-species log loss comparison")
print("=" * 60)

comparison_rows = []
for sp in species_list:
    mask = val_df[SPECIES_COL] == sp
    y_sp = val_df.loc[mask, TARGET_COL].values

    ll_single = log_loss(y_sp, single_val_probs[mask.values], labels=[0, 1])
    ll_multi = log_loss(y_sp, multi_val_probs[mask.values], labels=[0, 1])
    ll_int = log_loss(y_sp, int_val_probs[mask.values], labels=[0, 1])

    comparison_rows.append({
        "species": sp,
        "n_val": int(mask.sum()),
        "presence_rate": y_sp.mean(),
        "log_loss_single": ll_single,
        "log_loss_multi": ll_multi,
        "log_loss_multi_int": ll_int,
        "best_approach": min(
            [("single", ll_single), ("multi", ll_multi), ("multi_int", ll_int)],
            key=lambda x: x[1],
        )[0],
    })

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))
comparison_df.to_csv(RESULTS_PATH, index=False)
print(f"\nSaved per-species comparison to: {RESULTS_PATH}")

# ---------------------------------------------------------------------------
# 9. Overall summary
# ---------------------------------------------------------------------------
summary_df = pd.DataFrame([
    {"approach": "single_species", "log_loss": single_log_loss, "auc_roc": single_auc, "brier": single_brier},
    {"approach": "multi_species_onehot", "log_loss": multi_log_loss, "auc_roc": multi_auc, "brier": multi_brier},
    {"approach": "multi_species_interactions", "log_loss": int_log_loss, "auc_roc": int_auc, "brier": int_brier},
])
print("\n" + "=" * 60)
print("Overall summary")
print("=" * 60)
print(summary_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 10. Kaggle submissions
# ---------------------------------------------------------------------------
# 10a. Single-species submission
single_sub_probs = []
for sp in species_list:
    test_sp = test[test[SPECIES_COL] == sp].copy()
    X_te = build_single_species_X(test_sp)
    prob = np.clip(single_models[sp].predict_proba(X_te)[:, 1], 1e-6, 1 - 1e-6)
    test_sp["pred"] = prob
    single_sub_probs.append(test_sp[["id", "pred"]])

submission_single = pd.concat(single_sub_probs, ignore_index=True)
submission_single = submission_single.sort_values("id").reset_index(drop=True)
submission_single.to_csv(SUBMISSION_SINGLE_PATH, index=False)
print(f"\nSingle-species submission saved to: {SUBMISSION_SINGLE_PATH}")

# 10b. Multi-species submission
X_te_multi = build_multi_species_X(test, interactions=False)
multi_sub_probs = np.clip(multi_model.predict_proba(X_te_multi)[:, 1], 1e-6, 1 - 1e-6)
submission_multi = test[["id"]].copy()
submission_multi["pred"] = multi_sub_probs
submission_multi = submission_multi.sort_values("id").reset_index(drop=True)
submission_multi.to_csv(SUBMISSION_MULTI_PATH, index=False)
print(f"Multi-species submission saved to: {SUBMISSION_MULTI_PATH}")
