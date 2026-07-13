"""
GAM Model - Small Reptile Species Distribution (DATA5925)

This script follows Arun's updated modelling workflow

Workflow:
    1. Read train_split.csv and test_split.csv
    2. Build features
    3. Tune GAM parameters with 5-fold CV on train_split.csv
    4. Train final GAM on the full train_split.csv
    5. Evaluate on test_split.csv
    6. Report overall and per-species performance
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# 1. Configuration
# =========================

DATA_DIR = Path.cwd()

TRAIN_FILE = "train_split.csv"
TEST_FILE = "test_split.csv"
KAGGLE_TEST_FILE = "test.csv"
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"

KAGGLE_SUBMISSION_FILE = "gam_submission_0.csv"
TEST_METRICS_FILE = "test_metrics.csv"
PER_SPECIES_METRICS_FILE = "per_species_metrics.csv"

TARGET = "pres.abs"
SPECIES = "Species"
PLOT_COL = "plot"

CATEGORICAL_FEATURES = [
    "Species",
    "disturb",
    "soilfert",
]

SMOOTH_FEATURES = [
    "rainann",
    "soildepth",
    "tempann",
    "topo",
    "easting",
    "northing",
]

FEATURES = CATEGORICAL_FEATURES + SMOOTH_FEATURES

N_SPLITS = 5
CLIP_EPS = 1e-6
THRESHOLD = 0.5

PARAM_GRID = [
    {"n_knots": 4, "C": 0.1, "class_weight": None},
    {"n_knots": 4, "C": 0.1, "class_weight": "balanced"},
    {"n_knots": 4, "C": 0.5, "class_weight": None},
    {"n_knots": 4, "C": 0.5, "class_weight": "balanced"},
    {"n_knots": 6, "C": 0.1, "class_weight": None},
    {"n_knots": 6, "C": 0.1, "class_weight": "balanced"},
    {"n_knots": 6, "C": 0.5, "class_weight": None},
    {"n_knots": 6, "C": 0.5, "class_weight": "balanced"},
    {"n_knots": 8, "C": 0.1, "class_weight": None},
    {"n_knots": 8, "C": 0.1, "class_weight": "balanced"},
    {"n_knots": 8, "C": 0.5, "class_weight": None},
    {"n_knots": 8, "C": 0.5, "class_weight": "balanced"},
]


# =========================
# 2. Data loading
# =========================

def read_data():
    train_df = pd.read_csv(DATA_DIR / TRAIN_FILE)
    test_df = pd.read_csv(DATA_DIR / TEST_FILE)
    kaggle_test = pd.read_csv(DATA_DIR / KAGGLE_TEST_FILE)
    sample_submission = pd.read_csv(DATA_DIR / SAMPLE_SUBMISSION_FILE)
    return train_df, test_df, kaggle_test, sample_submission


def print_data_summary(train_df, test_df, kaggle_test):
    print(f"train: {len(train_df):>5} rows | presence rate {train_df[TARGET].mean():.4f}")
    print(f"test : {len(test_df):>5} rows | presence rate {test_df[TARGET].mean():.4f}")
    print(f"kaggle test: {len(kaggle_test):>5} rows")
    print(f"\n{len(FEATURES)} raw features: {FEATURES}")


def build_features(df):
    return df[FEATURES]


def build_target(df):
    return df[TARGET].astype(int).reset_index(drop=True)


# =========================
# 3. GAM model builder
# =========================

def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_gam_model(params):
    smooth_pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("spline", SplineTransformer(
            n_knots=params["n_knots"],
            degree=3,
            include_bias=False,
        )),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", make_onehot_encoder(), CATEGORICAL_FEATURES),
            ("smooth", smooth_pipeline, SMOOTH_FEATURES),
        ]
    )

    model = LogisticRegression(
        penalty="l2",
        C=params["C"],
        solver="lbfgs",
        max_iter=5000,
        class_weight=params["class_weight"],
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])


# =========================
# 4. Evaluation helpers
# =========================

def clip_prob(y_prob):
    return np.clip(np.asarray(y_prob, dtype=float), CLIP_EPS, 1 - CLIP_EPS)


def evaluate_binary(y_true, y_prob, threshold=THRESHOLD):
    y_true = np.asarray(y_true).astype(int)
    y_prob = clip_prob(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_test_metrics(metrics):
    print("\n" + "=" * 45)
    print("TEST PERFORMANCE")
    print("=" * 45)
    print(f"  Log loss (primary) : {metrics['log_loss']:.4f}")
    print(f"  ROC-AUC            : {metrics['auc']:.4f}")
    print(f"  PR-AUC (avg prec)  : {metrics['pr_auc']:.4f}")
    print(f"  Brier score        : {metrics['brier']:.4f}")
    print(f"  F1 (thr=0.5)       : {metrics['f1']:.4f}")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Sensitivity        : {metrics['sensitivity']:.4f}")
    print(f"  Specificity        : {metrics['specificity']:.4f}")
    print("\n  Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]))


# =========================
# 5. Cross-validation and tuning
# =========================

def cv_score(params, X, y, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    losses, aucs, briers = [], [], []

    print(f"\nTrying n_knots={params['n_knots']}, "
          f"C={params['C']}, class_weight={params['class_weight']}")

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = make_gam_model(params)
        model.fit(X_tr, y_tr)

        p_val = clip_prob(model.predict_proba(X_val)[:, 1])
        fold_metrics = evaluate_binary(y_val, p_val)

        losses.append(fold_metrics["log_loss"])
        aucs.append(fold_metrics["auc"])
        briers.append(fold_metrics["brier"])

        counts = y_val.value_counts().reindex([0, 1], fill_value=0)
        print(f"  Fold {fold}: negatives={counts[0]:>4}, positives={counts[1]:>3}, "
              f"logloss={fold_metrics['log_loss']:.4f}, "
              f"AUC={fold_metrics['auc']:.3f}, "
              f"Brier={fold_metrics['brier']:.4f}")

    return {
        **params,
        "mean_logloss": float(np.mean(losses)),
        "std_logloss": float(np.std(losses)),
        "mean_auc": float(np.nanmean(aucs)),
        "mean_brier": float(np.mean(briers)),
    }


def tune_hyperparameters(X_train, y_train, groups):
    print(f"\nGrid search over {len(PARAM_GRID)} GAM configurations, "
          f"{N_SPLITS}-fold CV on train_split.csv ...")

    results = []
    for i, params in enumerate(PARAM_GRID, 1):
        print(f"\n[{i:>2}/{len(PARAM_GRID)}]")
        score = cv_score(params, X_train, y_train, groups)
        results.append(score)
        print(f"  Mean logloss={score['mean_logloss']:.4f} "
              f"(+/-{score['std_logloss']:.4f}) | "
              f"Mean AUC={score['mean_auc']:.3f} | "
              f"Mean Brier={score['mean_brier']:.4f}")

    results_df = pd.DataFrame(results).sort_values("mean_logloss").reset_index(drop=True)

    print("\n" + "=" * 72)
    print("TUNING RESULTS BY MEAN CV LOG LOSS")
    print("=" * 72)
    display_cols = [
        "mean_logloss", "std_logloss", "mean_auc", "mean_brier",
        "n_knots", "C", "class_weight",
    ]
    print(results_df[display_cols].to_string(index=False))

    best = results_df.iloc[0]
    best_params = {
        "n_knots": int(best["n_knots"]),
        "C": float(best["C"]),
        "class_weight": None if pd.isna(best["class_weight"]) else best["class_weight"],
    }

    print("\nBest GAM parameters:")
    print(best_params)
    print(f"CV logloss = {best['mean_logloss']:.4f}")
    print(f"CV AUC     = {best['mean_auc']:.4f}")

    return best_params, results_df


# =========================
# 6. Final training and testing
# =========================

def train_final_model(X_train, y_train, best_params):
    final_model = make_gam_model(best_params)
    final_model.fit(X_train, y_train)
    return final_model


def evaluate_on_test(model, X_test, y_test):
    test_prob = clip_prob(model.predict_proba(X_test)[:, 1])
    metrics = evaluate_binary(y_test, test_prob)
    print_test_metrics(metrics)

    test_metrics = pd.DataFrame([metrics])
    test_metrics.to_csv(DATA_DIR / TEST_METRICS_FILE, index=False)

    print("\nTest metrics preview:")
    print(test_metrics)
    print(f"Saved: {TEST_METRICS_FILE}")

    return test_prob, metrics


def per_species_performance(test_df, y_test, test_prob):
    res = pd.DataFrame({
        "species": test_df[SPECIES].reset_index(drop=True),
        "y": y_test.reset_index(drop=True),
        "p": test_prob,
    })

    rows = []
    for sp, group in res.groupby("species"):
        m = evaluate_binary(group["y"], group["p"])
        rows.append({
            "species": sp,
            "n": m["n"],
            "positives": m["n_pos"],
            "prevalence": m["prevalence"],
            "log_loss": m["log_loss"],
            "brier": m["brier"],
            "auc": m["auc"],
            "f1": m["f1"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
        })

    per_species = (
        pd.DataFrame(rows)
        .sort_values("positives", ascending=False)
        .reset_index(drop=True)
    )

    print("\nPER-SPECIES PERFORMANCE")
    print(per_species.round(4).to_string(index=False))

    per_species.to_csv(DATA_DIR / PER_SPECIES_METRICS_FILE, index=False)

    print("\nPer-species metrics preview:")
    print(per_species.head())
    print(f"Saved: {PER_SPECIES_METRICS_FILE}")

    return per_species


def create_kaggle_submission(model, X_kaggle_test, sample_submission):
    kaggle_pred = model.predict_proba(X_kaggle_test)[:, 1]
    kaggle_pred = np.clip(kaggle_pred, 1e-6, 1 - 1e-6)

    submission = sample_submission.copy()
    submission["pred"] = kaggle_pred

    submission.to_csv(DATA_DIR / KAGGLE_SUBMISSION_FILE, index=False)

    print("\nKaggle submission preview:")
    print(submission.head())
    print(f"Saved: {KAGGLE_SUBMISSION_FILE}")

    return submission


# =========================
# 7. Main workflow
# =========================

def main():
    train_df, test_df, kaggle_test, sample_submission = read_data()
    print_data_summary(train_df, test_df, kaggle_test)

    X_train = build_features(train_df)
    y_train = build_target(train_df)
    groups = train_df[PLOT_COL].reset_index(drop=True)

    X_test = build_features(test_df)
    y_test = build_target(test_df)

    X_kaggle_test = build_features(kaggle_test)

    best_params, _ = tune_hyperparameters(X_train, y_train, groups)

    final_model = train_final_model(X_train, y_train, best_params)

    test_prob, _ = evaluate_on_test(final_model, X_test, y_test)
    per_species_performance(test_df, y_test, test_prob)
    create_kaggle_submission(final_model, X_kaggle_test, sample_submission)


if __name__ == "__main__":
    main()