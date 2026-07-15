"""
Single-species GAM baseline for DATA5925.

"Baseline" here means one independently tuned GAM per species. Every model:
  * reads the team train_split.csv and test_split.csv;
  * converts all eight numeric environmental predictors to cubic B-splines;
  * uses 5-fold stratified, plot-grouped cross-validation;
  * searches n_knots, C, and class_weight=None/'balanced';
  * selects hyperparameters by mean CV log loss;
  * reports Log Loss, AUC, Brier, F1, Sensitivity, and Specificity.

The reusable functions in this file are imported by gam_single_vs_multi.py.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train_split.csv"
TEST_PATH = ROOT / "test_split.csv"

TARGET = "pres.abs"
SPECIES = "Species"
GROUP = "plot"

ENV_FEATURES = [
    "easting",
    "northing",
    "disturb",
    "rainann",
    "soildepth",
    "soilfert",
    "tempann",
    "topo",
]
MODEL_FEATURES = [SPECIES] + ENV_FEATURES

N_SPLITS = 5
RANDOM_STATE = 42
THRESHOLD = 0.5
EPS = 1e-6

PARAM_GRID = [
    {"n_knots": n_knots, "C": C, "class_weight": class_weight}
    for n_knots in [4, 6, 8]
    for C in [0.1, 0.5]
    for class_weight in [None, "balanced"]
]


# ---------------------------------------------------------------------------
# 2. Data and feature construction
# ---------------------------------------------------------------------------

def read_split_data():
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


def build_xy(df):
    X = df[MODEL_FEATURES].reset_index(drop=True)
    y = df[TARGET].astype(int).reset_index(drop=True)
    groups = df[GROUP].reset_index(drop=True)
    return X, y, groups


def make_onehot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse=False,
        )


class GAMFeatureTransformer(BaseEstimator, TransformerMixin):
    """Shared spline basis with optional Species effects/interactions."""

    def __init__(self, n_knots=6, include_species=False, interactions=False):
        self.n_knots = n_knots
        self.include_species = include_species
        self.interactions = interactions

    def fit(self, X, y=None):
        env = X[ENV_FEATURES]
        self.scaler_ = StandardScaler().fit(env)
        scaled = self.scaler_.transform(env)
        self.spline_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=3,
            include_bias=False,
        ).fit(scaled)

        if self.include_species:
            self.species_encoder_ = make_onehot_encoder()
            self.species_encoder_.fit(X[[SPECIES]])
        return self

    def transform(self, X):
        scaled = self.scaler_.transform(X[ENV_FEATURES])
        smooth = self.spline_.transform(scaled)

        if not self.include_species:
            return smooth

        species = self.species_encoder_.transform(X[[SPECIES]])
        parts = [smooth, species]

        if self.interactions:
            # Each non-reference Species receives a deviation for every spline
            # basis column: Species dummy x s(environmental predictor).
            smooth_deviations = (
                species[:, :, np.newaxis] * smooth[:, np.newaxis, :]
            ).reshape(len(X), -1)
            parts.append(smooth_deviations)

        return np.hstack(parts)


def make_gam(params, include_species=False, interactions=False):
    return Pipeline([
        (
            "features",
            GAMFeatureTransformer(
                n_knots=params["n_knots"],
                include_species=include_species,
                interactions=interactions,
            ),
        ),
        (
            "model",
            LogisticRegression(
                C=params["C"],
                solver="lbfgs",
                max_iter=5000,
                class_weight=params["class_weight"],
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------

def clip_probability(probability):
    return np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)


def evaluate_binary(y_true, probability):
    y_true = np.asarray(y_true, dtype=int)
    probability = clip_probability(probability)
    prediction = (probability >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "n": len(y_true),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "log_loss": log_loss(y_true, probability, labels=[0, 1]),
        "auc": roc_auc_score(y_true, probability)
        if np.unique(y_true).size == 2 else np.nan,
        "brier": brier_score_loss(y_true, probability),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
    }


# ---------------------------------------------------------------------------
# 4. Five-fold plot-level tuning
# ---------------------------------------------------------------------------

def make_cv():
    return StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def score_params(params, X, y, groups, include_species, interactions):
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(
        make_cv().split(X, y, groups),
        1,
    ):
        model = make_gam(params, include_species, interactions)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probability = model.predict_proba(X.iloc[val_idx])[:, 1]
        metrics = evaluate_binary(y.iloc[val_idx], probability)
        fold_rows.append({"fold": fold, **metrics})

    folds = pd.DataFrame(fold_rows)
    return {
        **params,
        "mean_log_loss": folds["log_loss"].mean(),
        "std_log_loss": folds["log_loss"].std(ddof=0),
        "mean_auc": folds["auc"].mean(),
        "mean_brier": folds["brier"].mean(),
    }


def tune_gam(train_df, label, include_species=False, interactions=False):
    X, y, groups = build_xy(train_df)
    rows = []

    print(f"\nTuning {label}: {len(PARAM_GRID)} configs x {N_SPLITS} folds")
    for params in PARAM_GRID:
        result = score_params(
            params,
            X,
            y,
            groups,
            include_species,
            interactions,
        )
        rows.append(result)
        print(
            f"  knots={params['n_knots']} C={params['C']} "
            f"weight={str(params['class_weight']):>8} | "
            f"logloss={result['mean_log_loss']:.4f} "
            f"AUC={result['mean_auc']:.4f}"
        )

    results = pd.DataFrame(rows).sort_values("mean_log_loss").reset_index(drop=True)
    best = results.iloc[0]
    best_params = {
        "n_knots": int(best["n_knots"]),
        "C": float(best["C"]),
        "class_weight": None
        if pd.isna(best["class_weight"]) else best["class_weight"],
    }
    print(f"Best {label}: {best_params}")
    return best_params, results


# ---------------------------------------------------------------------------
# 5. Reusable single- and multi-species runners
# ---------------------------------------------------------------------------

def run_single_species(train_df, test_df):
    test_probability = np.zeros(len(test_df), dtype=float)
    models = {}
    selected_params = {}
    cv_tables = []

    for species in sorted(train_df[SPECIES].unique()):
        train_sp = train_df[train_df[SPECIES] == species].reset_index(drop=True)
        test_mask = test_df[SPECIES] == species
        test_sp = test_df.loc[test_mask].reset_index(drop=True)

        params, cv_results = tune_gam(train_sp, f"single | {species}")
        cv_results.insert(0, "species", species)
        cv_results.insert(0, "approach", "single_species")
        cv_tables.append(cv_results)

        X_train, y_train, _ = build_xy(train_sp)
        X_test, _, _ = build_xy(test_sp)
        model = make_gam(params)
        model.fit(X_train, y_train)
        test_probability[test_mask.to_numpy()] = clip_probability(
            model.predict_proba(X_test)[:, 1]
        )
        models[species] = model
        selected_params[species] = params

    return {
        "models": models,
        "params": selected_params,
        "probability": test_probability,
        "cv_results": pd.concat(cv_tables, ignore_index=True),
    }


def run_multi_species(train_df, test_df, interactions=False):
    approach = "multi_interactions" if interactions else "multi_species"
    params, cv_results = tune_gam(
        train_df,
        approach,
        include_species=True,
        interactions=interactions,
    )
    cv_results.insert(0, "species", "ALL")
    cv_results.insert(0, "approach", approach)

    X_train, y_train, _ = build_xy(train_df)
    X_test, _, _ = build_xy(test_df)
    model = make_gam(
        params,
        include_species=True,
        interactions=interactions,
    )
    model.fit(X_train, y_train)
    probability = clip_probability(model.predict_proba(X_test)[:, 1])
    return {
        "model": model,
        "params": params,
        "probability": probability,
        "cv_results": cv_results,
    }


def make_per_species_table(test_df, probability, approach):
    rows = []
    scored = test_df.assign(_probability=probability)
    for species, group in scored.groupby(SPECIES):
        rows.append({
            "approach": approach,
            "species": species,
            **evaluate_binary(group[TARGET], group["_probability"]),
        })
    return pd.DataFrame(rows)


def selected_params_table(params, approach):
    if approach == "single_species":
        return pd.DataFrame([
            {"approach": approach, "species": species, **values}
            for species, values in params.items()
        ])
    return pd.DataFrame([
        {"approach": approach, "species": "ALL", **params}
    ])


# ---------------------------------------------------------------------------
# 6. Baseline output
# ---------------------------------------------------------------------------

def main():
    train_df, test_df = read_split_data()
    print("Single-species GAM baseline")
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"Spline predictors: {ENV_FEATURES}")

    result = run_single_species(train_df, test_df)
    overall = pd.DataFrame([{
        "approach": "single_species",
        **evaluate_binary(test_df[TARGET], result["probability"]),
    }])
    per_species = make_per_species_table(
        test_df,
        result["probability"],
        "single_species",
    )
    selected = selected_params_table(result["params"], "single_species")

    overall.to_csv(ROOT / "gam_baseline_overall.csv", index=False)
    per_species.to_csv(ROOT / "gam_baseline_per_species.csv", index=False)
    selected.to_csv(ROOT / "gam_baseline_selected_params.csv", index=False)
    result["cv_results"].to_csv(ROOT / "gam_baseline_cv_results.csv", index=False)

    print("\nOverall test performance")
    print(overall.round(4).to_string(index=False))
    print("\nPer-species test performance")
    print(per_species.round(4).to_string(index=False))
    print("\nSaved baseline CSV files in:", ROOT)
    return result


if __name__ == "__main__":
    BASELINE_RESULTS = main()
