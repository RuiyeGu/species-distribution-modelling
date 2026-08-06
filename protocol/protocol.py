"""Reusable Task 3 protocol for all model families."""

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
