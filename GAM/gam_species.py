"""
GAM models for the DATA5925 small-reptile species distribution project.

The normal workflow uses only train_split.csv and test_split.csv:
    1. Single-species GAMs: one model per Species.
    2. Pooled multi-species GAM: shared smooths plus Species main effects.
    3. Interactive multi-species GAM: shared smooths plus
       Species-specific smooth deviations.

All models use 5-fold plot-level cross-validation and are selected by mean
CV log loss. Kaggle files are read only when RUN_KAGGLE is set to True or
create_kaggle_submission(...) is called explicitly.
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


# =========================
# 1. Configuration
# =========================

DATA_DIR = Path(__file__).resolve().parent

TRAIN_FILE = "train_split.csv"
TEST_FILE = "test_split.csv"
KAGGLE_TEST_FILE = "test.csv"
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"

OVERALL_METRICS_FILE = "gam_overall_metrics.csv"
PER_SPECIES_METRICS_FILE = "gam_per_species_metrics.csv"

TARGET = "pres.abs"
SPECIES = "Species"
PLOT_COL = "plot"

# All eight environmental predictors are numeric and receive cubic splines.
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
CLIP_EPS = 1e-6

PARAM_GRID = [
    {"n_knots": n_knots, "C": C, "class_weight": class_weight}
    for n_knots in [4, 6, 8]
    for C in [0.1, 0.5]
    for class_weight in [None, "balanced"]
]

RUN_KAGGLE = False
KAGGLE_APPROACH = "multi_species_interactions"


# =========================
# 2. Data loading
# =========================

def read_split_data():
    """Read only the labelled files used for training and evaluation."""
    train_df = pd.read_csv(DATA_DIR / TRAIN_FILE)
    test_df = pd.read_csv(DATA_DIR / TEST_FILE)
    return train_df, test_df


def read_kaggle_data():
    """Read Kaggle files only when a submission is requested."""
    kaggle_test = pd.read_csv(DATA_DIR / KAGGLE_TEST_FILE)
    sample_submission = pd.read_csv(DATA_DIR / SAMPLE_SUBMISSION_FILE)
    return kaggle_test, sample_submission


def build_xy(df):
    X = df[MODEL_FEATURES].reset_index(drop=True)
    y = df[TARGET].astype(int).reset_index(drop=True)
    groups = df[PLOT_COL].reset_index(drop=True)
    return X, y, groups


# =========================
# 3. GAM feature construction
# =========================

def make_onehot_encoder(drop=None):
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            drop=drop,
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            drop=drop,
            sparse=False,
        )


class GAMFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create shared spline terms and optional Species-specific smooths."""

    def __init__(self, n_knots=6, include_species=True, interactions=False):
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
            # The omitted Species is the reference. Shared splines describe its
            # curve; interaction columns describe each other Species' deviation.
            self.species_encoder_ = make_onehot_encoder(drop="first")
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
            # Species dummy x every spline basis column gives a true
            # Species x environmental smooth interaction.
            interaction = (
                species[:, :, np.newaxis] * smooth[:, np.newaxis, :]
            ).reshape(len(X), -1)
            parts.append(interaction)

        return np.hstack(parts)


def make_gam_model(params, include_species=True, interactions=False):
    features = GAMFeatureTransformer(
        n_knots=params["n_knots"],
        include_species=include_species,
        interactions=interactions,
    )
    classifier = LogisticRegression(
        C=params["C"],
        solver="lbfgs",
        max_iter=5000,
        class_weight=params["class_weight"],
    )
    return Pipeline([
        ("features", features),
        ("model", classifier),
    ])


# =========================
# 4. Metrics
# =========================

def clip_prob(y_prob):
    return np.clip(np.asarray(y_prob, dtype=float), CLIP_EPS, 1 - CLIP_EPS)


def evaluate_binary(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = clip_prob(y_prob)
    y_pred = (y_prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "n": len(y_true),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
        "auc": roc_auc_score(y_true, y_prob)
        if np.unique(y_true).size == 2 else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
    }


# =========================
# 5. Five-fold CV and tuning
# =========================

def make_cv():
    return StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def cv_score(params, X, y, groups, include_species, interactions):
    losses, aucs, briers = [], [], []

    for train_idx, val_idx in make_cv().split(X, y, groups):
        model = make_gam_model(params, include_species, interactions)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = clip_prob(model.predict_proba(X.iloc[val_idx])[:, 1])
        metrics = evaluate_binary(y.iloc[val_idx], prob)
        losses.append(metrics["log_loss"])
        aucs.append(metrics["auc"])
        briers.append(metrics["brier"])

    return {
        **params,
        "mean_log_loss": float(np.mean(losses)),
        "std_log_loss": float(np.std(losses)),
        "mean_auc": float(np.nanmean(aucs)),
        "mean_brier": float(np.mean(briers)),
    }


def tune_model(train_df, label, include_species, interactions):
    X, y, groups = build_xy(train_df)
    rows = []

    print(f"\nTuning {label}: {len(PARAM_GRID)} configurations, {N_SPLITS}-fold CV")
    for params in PARAM_GRID:
        score = cv_score(
            params,
            X,
            y,
            groups,
            include_species,
            interactions,
        )
        rows.append(score)
        print(
            f"  knots={params['n_knots']} C={params['C']} "
            f"weight={str(params['class_weight']):>8} | "
            f"logloss={score['mean_log_loss']:.4f} "
            f"AUC={score['mean_auc']:.4f}"
        )

    best = min(rows, key=lambda row: row["mean_log_loss"])
    best_params = {
        "n_knots": best["n_knots"],
        "C": best["C"],
        "class_weight": best["class_weight"],
    }
    print(f"Best {label}: {best_params}, CV logloss={best['mean_log_loss']:.4f}")
    return best_params


# =========================
# 6. Single-species GAMs
# =========================

def run_single_species(train_df, test_df):
    models = {}
    best_params = {}
    test_prob = np.zeros(len(test_df), dtype=float)

    print("\n" + "=" * 64)
    print("SINGLE-SPECIES GAM: one model per Species")
    print("=" * 64)

    for species in sorted(train_df[SPECIES].unique()):
        train_sp = train_df[train_df[SPECIES] == species].reset_index(drop=True)
        test_mask = test_df[SPECIES] == species
        test_sp = test_df.loc[test_mask].reset_index(drop=True)

        params = tune_model(
            train_sp,
            f"single | {species}",
            include_species=False,
            interactions=False,
        )
        X_train, y_train, _ = build_xy(train_sp)
        model = make_gam_model(params, include_species=False, interactions=False)
        model.fit(X_train, y_train)

        X_test, _, _ = build_xy(test_sp)
        test_prob[test_mask.to_numpy()] = clip_prob(
            model.predict_proba(X_test)[:, 1]
        )
        models[species] = model
        best_params[species] = params

    return models, best_params, test_prob


# =========================
# 7. Multi-species GAMs
# =========================

def run_multi_species(train_df, test_df, interactions=False):
    approach = (
        "multi_species_interactions" if interactions else "multi_species"
    )
    title = (
        "MULTI-SPECIES GAM: Species-specific smooth interactions"
        if interactions else
        "MULTI-SPECIES GAM: shared smooths and Species effects"
    )
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)

    params = tune_model(
        train_df,
        approach,
        include_species=True,
        interactions=interactions,
    )
    X_train, y_train, _ = build_xy(train_df)
    X_test, _, _ = build_xy(test_df)
    model = make_gam_model(params, include_species=True, interactions=interactions)
    model.fit(X_train, y_train)
    test_prob = clip_prob(model.predict_proba(X_test)[:, 1])
    return model, params, test_prob


# =========================
# 8. Test-set comparison
# =========================

def compare_on_test(test_df, predictions):
    y_test = test_df[TARGET].astype(int).reset_index(drop=True)
    overall_rows = []
    species_rows = []

    for approach, prob in predictions.items():
        overall_rows.append({
            "approach": approach,
            **evaluate_binary(y_test, prob),
        })

        for species, group in test_df.assign(_prob=prob).groupby(SPECIES):
            species_rows.append({
                "approach": approach,
                "species": species,
                **evaluate_binary(group[TARGET], group["_prob"]),
            })

    overall = pd.DataFrame(overall_rows).sort_values("log_loss")
    per_species = pd.DataFrame(species_rows).sort_values(
        ["species", "log_loss"]
    )

    print("\n" + "=" * 64)
    print("HELD-OUT TEST-SPLIT PERFORMANCE")
    print("=" * 64)
    print(overall.round(4).to_string(index=False))
    print("\nPER-SPECIES PERFORMANCE")
    print(per_species.round(4).to_string(index=False))

    overall.to_csv(DATA_DIR / OVERALL_METRICS_FILE, index=False)
    per_species.to_csv(DATA_DIR / PER_SPECIES_METRICS_FILE, index=False)
    print(f"\nSaved: {OVERALL_METRICS_FILE}")
    print(f"Saved: {PER_SPECIES_METRICS_FILE}")
    return overall, per_species


# =========================
# 9. Optional Kaggle submission
# =========================

def create_kaggle_submission(best_params, approach=KAGGLE_APPROACH):
    """Refit one chosen approach on all labelled rows and predict Kaggle test."""
    train_df, test_df = read_split_data()
    full_train = pd.concat([train_df, test_df], ignore_index=True)
    kaggle_test, sample_submission = read_kaggle_data()
    kaggle_prob = np.zeros(len(kaggle_test), dtype=float)

    if approach == "single_species":
        for species in sorted(full_train[SPECIES].unique()):
            train_sp = full_train[full_train[SPECIES] == species].reset_index(drop=True)
            test_mask = kaggle_test[SPECIES] == species
            test_sp = kaggle_test.loc[test_mask].reset_index(drop=True)
            X_train, y_train, _ = build_xy(train_sp)
            model = make_gam_model(
                best_params[approach][species],
                include_species=False,
                interactions=False,
            )
            model.fit(X_train, y_train)
            X_kaggle = test_sp[MODEL_FEATURES].reset_index(drop=True)
            kaggle_prob[test_mask.to_numpy()] = clip_prob(
                model.predict_proba(X_kaggle)[:, 1]
            )
    else:
        interactions = approach == "multi_species_interactions"
        X_train, y_train, _ = build_xy(full_train)
        model = make_gam_model(
            best_params[approach],
            include_species=True,
            interactions=interactions,
        )
        model.fit(X_train, y_train)
        X_kaggle = kaggle_test[MODEL_FEATURES].reset_index(drop=True)
        kaggle_prob = clip_prob(model.predict_proba(X_kaggle)[:, 1])

    submission = sample_submission.copy()
    submission["pred"] = kaggle_prob
    output_file = f"gam_submission_{approach}.csv"
    submission.to_csv(DATA_DIR / output_file, index=False)
    print(f"\nSaved Kaggle submission: {output_file}")
    print(submission.head())
    return submission


# =========================
# 10. Main workflow
# =========================

def main():
    train_df, test_df = read_split_data()
    print(f"Train split: {len(train_df)} rows, {train_df[PLOT_COL].nunique()} plots")
    print(f"Test split : {len(test_df)} rows, {test_df[PLOT_COL].nunique()} plots")
    print(f"Environmental spline variables: {ENV_FEATURES}")

    single_models, single_params, single_prob = run_single_species(
        train_df,
        test_df,
    )
    pooled_model, pooled_params, pooled_prob = run_multi_species(
        train_df,
        test_df,
        interactions=False,
    )
    interactive_model, interactive_params, interactive_prob = run_multi_species(
        train_df,
        test_df,
        interactions=True,
    )

    predictions = {
        "single_species": single_prob,
        "multi_species": pooled_prob,
        "multi_species_interactions": interactive_prob,
    }
    overall, per_species = compare_on_test(test_df, predictions)

    best_params = {
        "single_species": single_params,
        "multi_species": pooled_params,
        "multi_species_interactions": interactive_params,
    }
    results = {
        "models": {
            "single_species": single_models,
            "multi_species": pooled_model,
            "multi_species_interactions": interactive_model,
        },
        "best_params": best_params,
        "overall_metrics": overall,
        "per_species_metrics": per_species,
    }

    if RUN_KAGGLE:
        create_kaggle_submission(best_params, KAGGLE_APPROACH)

    return results


if __name__ == "__main__":
    RESULTS = main()

    # Alternative one-line submission after training:
    # create_kaggle_submission(RESULTS["best_params"], "multi_species_interactions")
