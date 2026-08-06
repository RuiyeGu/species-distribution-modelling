"""Reusable protocol for all model families."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


PROTOCOL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROTOCOL_DIR / "protocol_config.json"


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_split(repo_root, split_type):
    """Load and validate one of the fixed Task 1/2 plot-level splits."""
    config = load_config()
    if split_type not in config["splits"]:
        raise ValueError(f"Unknown split_type: {split_type}")

    repo_root = Path(repo_root)
    paths = config["splits"][split_type]
    train = pd.read_csv(repo_root / paths["train"])
    test = pd.read_csv(repo_root / paths["test"])

    required = set(config["predictors"] + [
        config["target"], config["species"], config["group"], config["id"]
    ])
    for label, frame in [("train", train), ("test", test)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} data is missing columns: {sorted(missing)}")

    group = config["group"]
    overlap = set(train[group]) & set(test[group])
    if overlap:
        raise ValueError(f"Plot leakage detected: {len(overlap)} shared plots")
    return train, test


def make_preprocessor():
    """Create the common preprocessing fitted only on each training set."""
    config = load_config()
    predictors = config["predictors"]
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        [("numeric", numeric, predictors)],
        remainder="drop",
    )


def evaluate(y_true, probability):
    """Calculate the common Task 3 evaluation metrics."""
    config = load_config()
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    prediction = (probability >= config["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    two_classes = np.unique(y_true).size == 2

    return {
        "n": len(y_true),
        "positives": int(y_true.sum()),
        "log_loss": log_loss(y_true, probability, labels=[0, 1]),
        "auc": roc_auc_score(y_true, probability) if two_classes else np.nan,
        "average_precision": average_precision_score(y_true, probability)
        if two_classes else np.nan,
        "brier": brier_score_loss(y_true, probability),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
    }


def run_experiment(model_name, estimator, split_type, repo_root, output_dir):
    """Fit one cloned estimator per species and save standard outputs."""
    config = load_config()
    predictors = config["predictors"]
    target = config["target"]
    species_col = config["species"]
    group = config["group"]
    id_col = config["id"]
    train, test = load_split(repo_root, split_type)

    prediction_tables = []
    metric_rows = []
    for species in sorted(train[species_col].unique()):
        train_sp = train[train[species_col] == species].copy()
        test_sp = test[test[species_col] == species].copy()
        pipeline = Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", clone(estimator)),
        ])
        pipeline.fit(train_sp[predictors], train_sp[target].astype(int))
        probability = np.clip(
            pipeline.predict_proba(test_sp[predictors])[:, 1], 1e-6, 1 - 1e-6
        )

        metric_rows.append({
            "model": model_name,
            "split_type": split_type,
            "species": species,
            **evaluate(test_sp[target], probability),
        })
        prediction_tables.append(pd.DataFrame({
            "plot": test_sp[group].to_numpy(),
            "id": test_sp[id_col].to_numpy(),
            "species": species,
            "model": model_name,
            "split_type": split_type,
            "y_true": test_sp[target].astype(int).to_numpy(),
            "y_prob": probability,
            "easting": test_sp["easting"].to_numpy(),
            "northing": test_sp["northing"].to_numpy(),
        }))

    predictions = pd.concat(prediction_tables, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / f"{model_name}_{split_type}_predictions.csv", index=False)
    metrics.to_csv(output_dir / f"{model_name}_{split_type}_metrics.csv", index=False)
    return predictions, metrics


def run_tuned_experiment(
    model_name,
    estimator,
    param_grid,
    split_type,
    repo_root,
    output_dir,
    n_splits=None,
    selection_metric=None,
):
    """Tune each species on training plots only, then test exactly once.

    Parameter names refer to the complete pipeline. For example, when the
    supplied estimator is a Pipeline with ``splines`` and ``classifier`` steps,
    use ``model__splines__n_knots`` and ``model__classifier__C``.
    """
    config = load_config()
    tuning = config["tuning"]
    n_splits = tuning["n_splits"] if n_splits is None else n_splits
    selection_metric = (
        tuning["selection_metric"]
        if selection_metric is None
        else selection_metric
    )
    if tuning["scope"] != "training_only":
        raise ValueError("Tuning scope must be training_only")
    if tuning["cv_method"] != "stratified_group_k_fold":
        raise ValueError("Unsupported shared CV method")
    if selection_metric != "log_loss" or tuning["selection_direction"] != "minimize":
        raise ValueError("The shared protocol selects by minimum log_loss")
    if not tuning["refit_on_full_training_set"] or not tuning["evaluate_test_once"]:
        raise ValueError("Task 5 requires full-train refit and one final test evaluation")

    predictors = config["predictors"]
    target = config["target"]
    species_col = config["species"]
    group_col = config["group"]
    id_col = config["id"]
    seed = config["random_seed"]
    train, test = load_split(repo_root, split_type)

    prediction_tables = []
    metric_rows = []
    cv_rows = []
    best_param_rows = []

    for species in sorted(train[species_col].unique()):
        train_sp = train[train[species_col] == species].reset_index(drop=True)
        test_sp = test[test[species_col] == species].reset_index(drop=True)
        X_train = train_sp[predictors]
        y_train = train_sp[target].astype(int)
        groups = train_sp[group_col]

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        candidates = []

        for candidate_number, params in enumerate(ParameterGrid(param_grid), start=1):
            fold_losses = []
            for fold, (fit_index, validation_index) in enumerate(
                splitter.split(X_train, y_train, groups), start=1
            ):
                pipeline = Pipeline([
                    ("preprocess", make_preprocessor()),
                    ("model", clone(estimator)),
                ])
                pipeline.set_params(**params)
                pipeline.fit(X_train.iloc[fit_index], y_train.iloc[fit_index])
                probability = np.clip(
                    pipeline.predict_proba(X_train.iloc[validation_index])[:, 1],
                    1e-6,
                    1 - 1e-6,
                )
                fold_metrics = evaluate(y_train.iloc[validation_index], probability)
                fold_losses.append(fold_metrics[selection_metric])
                cv_rows.append({
                    "model": model_name,
                    "split_type": split_type,
                    "species": species,
                    "candidate": candidate_number,
                    "fold": fold,
                    **params,
                    **fold_metrics,
                })

            candidates.append({
                "params": params,
                "mean_log_loss": float(np.mean(fold_losses)),
                "std_log_loss": float(np.std(fold_losses)),
            })

        best = min(candidates, key=lambda row: row["mean_log_loss"])
        best_params = best["params"]
        best_param_rows.append({
            "model": model_name,
            "split_type": split_type,
            "species": species,
            **best_params,
            "mean_cv_log_loss": best["mean_log_loss"],
            "std_cv_log_loss": best["std_log_loss"],
        })

        final_pipeline = Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", clone(estimator)),
        ])
        final_pipeline.set_params(**best_params)
        final_pipeline.fit(X_train, y_train)
        probability = np.clip(
            final_pipeline.predict_proba(test_sp[predictors])[:, 1],
            1e-6,
            1 - 1e-6,
        )

        metric_rows.append({
            "model": model_name,
            "split_type": split_type,
            "species": species,
            **evaluate(test_sp[target], probability),
        })
        prediction_tables.append(pd.DataFrame({
            "plot": test_sp[group_col].to_numpy(),
            "id": test_sp[id_col].to_numpy(),
            "species": species,
            "model": model_name,
            "split_type": split_type,
            "y_true": test_sp[target].astype(int).to_numpy(),
            "y_prob": probability,
            "easting": test_sp["easting"].to_numpy(),
            "northing": test_sp["northing"].to_numpy(),
        }))

    predictions = pd.concat(prediction_tables, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    cv_results = pd.DataFrame(cv_rows)
    best_params = pd.DataFrame(best_param_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_name}_{split_type}"
    predictions.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    metrics.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
    cv_results.to_csv(output_dir / f"{prefix}_cv_results.csv", index=False)
    best_params.to_csv(output_dir / f"{prefix}_best_params.csv", index=False)
    return predictions, metrics, cv_results, best_params
