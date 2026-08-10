"""MLP runner for the shared evaluation protocol.

Mirrors ``Random Forest/random_forest.py`` so that the MLP is evaluated on the
same plot-level splits, the same tune-on-train / test-once design, and the same
metric suite as the other four model families.

Two MLP-specific choices (see thesis Chapter 4):

1. No class weighting. scikit-learn's ``MLPClassifier`` does not support class
   weights, and the thesis shows that weighting inflates log loss while leaving
   AUC unchanged. The network therefore optimises *unweighted* cross-entropy,
   and imbalance is handled for threshold-dependent metrics by tuning the
   decision threshold on out-of-fold predictions (``tune_threshold`` below)
   rather than by reweighting the loss.

2. Standardised inputs. Neural networks are scale-sensitive; the shared
   preprocessor already imputes (median) and standardises, fitted on the
   training fold only, so no held-out information leaks into the transform.

Run (from inside this folder), pointing --repo-root at the split CSVs:

    python MLP.py --repo-root "../../Split data/dataset"

Set USE_COORDS below to False to produce the no-coordinate variant.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# protocol.py lives one directory up (protocol/); make it importable regardless
# of where this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protocol import (
    evaluate,
    get_predictors,
    load_config,
    load_split,
    run_tuned_experiment,
)


CONFIG = load_config()
# Match the RF convention: change this one line to switch coordinate inputs.
USE_COORDS = True
ENV_FEATURES = get_predictors(use_coords=USE_COORDS)
TARGET = CONFIG["target"]
SPECIES = CONFIG["species"]
GROUP = CONFIG["group"]
SEED = CONFIG["random_seed"]
N_SPLITS = CONFIG["tuning"]["n_splits"]

# Grid mirrors the thesis MLP search: small architectures suited to few
# positives, and a strong-penalty range because with ~500 training plots and as
# few as three positives for the rarest species, heavy regularisation is
# expected to be necessary. early_stopping is left off (it would carve out a
# random validation subset and break plot-level grouping); regularisation is
# controlled entirely by alpha, selected under the shared grouped CV.
MLP_PARAM_GRID = {
    "model__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
    "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    "model__learning_rate_init": [1e-2, 1e-3],
}


def make_single_mlp():
    """Single-species MLP: one network per species, unweighted cross-entropy."""
    return MLPClassifier(
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=False,
        random_state=SEED,
    )


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


class MultiMLPTransformer(BaseEstimator, TransformerMixin):
    """Pool all species; append a one-hot Species indicator.

    ``interactions=True`` additionally appends Species-by-environment products,
    giving the pooled network species-specific environmental responses (the
    ``multi_interactions`` formulation). This matches the pooling mechanism used
    by the tree/regression families.
    """

    def __init__(self, interactions=False):
        self.interactions = interactions

    def fit(self, X, y=None):
        self.imputer_ = SimpleImputer(strategy="median")
        imputed = self.imputer_.fit_transform(X[ENV_FEATURES])
        self.scaler_ = StandardScaler()
        self.scaler_.fit(imputed)
        self.species_ = one_hot_encoder()
        self.species_.fit(X[[SPECIES]])
        return self

    def transform(self, X):
        imputed = self.imputer_.transform(X[ENV_FEATURES])
        environment = self.scaler_.transform(imputed)
        species = self.species_.transform(X[[SPECIES]])
        parts = [environment, species]
        if self.interactions:
            interaction = (
                species[:, :, np.newaxis] * environment[:, np.newaxis, :]
            ).reshape(len(X), -1)
            parts.append(interaction)
        return np.hstack(parts)


def make_multi_mlp(interactions=False):
    return Pipeline([
        ("features", MultiMLPTransformer(interactions=interactions)),
        ("classifier", MLPClassifier(
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=False,
            random_state=SEED,
        )),
    ])


def tune_multi(train, test, split_type, approach, interactions, output_dir):
    """Tune a pooled MLP on training plots only, then test exactly once.

    Structurally identical to the RF runner's ``tune_multi`` so the outputs line
    up column-for-column with the other families.
    """
    model = make_multi_mlp(interactions)
    grid = {
        "classifier__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
        "classifier__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "classifier__learning_rate_init": [1e-2, 1e-3],
    }

    X_train = train[[SPECIES] + ENV_FEATURES].reset_index(drop=True)
    y_train = train[TARGET].astype(int).reset_index(drop=True)
    groups = train[GROUP].reset_index(drop=True)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED
    )
    cv_rows = []
    candidates = []

    for candidate, params in enumerate(ParameterGrid(grid), start=1):
        losses = []
        for fold, (fit_idx, val_idx) in enumerate(
            splitter.split(X_train, y_train, groups), start=1
        ):
            fitted = clone(model).set_params(**params)
            fitted.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
            probability = fitted.predict_proba(X_train.iloc[val_idx])[:, 1]
            metrics = evaluate(y_train.iloc[val_idx], probability)
            losses.append(metrics["log_loss"])
            cv_rows.append({
                "model": approach,
                "split_type": split_type,
                "use_coords": USE_COORDS,
                "candidate": candidate,
                "fold": fold,
                **params,
                **metrics,
            })
        candidates.append({
            "params": params,
            "mean_cv_log_loss": float(np.mean(losses)),
            "std_cv_log_loss": float(np.std(losses)),
        })

    best = min(candidates, key=lambda row: row["mean_cv_log_loss"])
    final_model = clone(model).set_params(**best["params"])
    final_model.fit(X_train, y_train)
    probability = np.clip(
        final_model.predict_proba(test[[SPECIES] + ENV_FEATURES])[:, 1],
        1e-6,
        1 - 1e-6,
    )

    predictions = pd.DataFrame({
        "plot": test[GROUP].to_numpy(),
        "id": test[CONFIG["id"]].to_numpy(),
        "species": test[SPECIES].to_numpy(),
        "model": approach,
        "split_type": split_type,
        "use_coords": USE_COORDS,
        "y_true": test[TARGET].astype(int).to_numpy(),
        "y_prob": probability,
        "easting": test["easting"].to_numpy(),
        "northing": test["northing"].to_numpy(),
    })
    metric_rows = []
    for species, group in predictions.groupby("species"):
        metric_rows.append({
            "model": approach,
            "split_type": split_type,
            "use_coords": USE_COORDS,
            "species": species,
            **evaluate(group["y_true"], group["y_prob"]),
        })
    metrics = pd.DataFrame(metric_rows)
    best_params = pd.DataFrame([{
        "model": approach,
        "split_type": split_type,
        "use_coords": USE_COORDS,
        **best["params"],
        "mean_cv_log_loss": best["mean_cv_log_loss"],
        "std_cv_log_loss": best["std_cv_log_loss"],
    }])
    cv_results = pd.DataFrame(cv_rows)
    prefix = f"{approach}_{split_type}"
    predictions.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    metrics.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
    cv_results.to_csv(output_dir / f"{prefix}_cv_results.csv", index=False)
    best_params.to_csv(output_dir / f"{prefix}_best_params.csv", index=False)
    return predictions, metrics, cv_results, best_params


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=script_dir)
    parser.add_argument("--output-dir", type=Path, default=script_dir / "Output")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    all_metrics = []
    coordinate_label = "with_coords" if USE_COORDS else "no_coords"
    for split_type in ["spatial", "random"]:
        train, test = load_split(args.repo_root, split_type)

        single_predictions, single_metrics, _, _ = run_tuned_experiment(
            model_name=f"MLP_single_{coordinate_label}",
            estimator=make_single_mlp(),
            param_grid=MLP_PARAM_GRID,
            split_type=split_type,
            repo_root=args.repo_root,
            output_dir=output_dir,
            use_coords=USE_COORDS,
        )
        all_predictions.append(single_predictions)
        all_metrics.append(single_metrics)

        for approach, interactions in [
            (f"MLP_multi_{coordinate_label}", False),
            (f"MLP_multi_interactions_{coordinate_label}", True),
        ]:
            predictions, metrics, _, _ = tune_multi(
                train, test, split_type, approach, interactions, output_dir
            )
            all_predictions.append(predictions)
            all_metrics.append(metrics)

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions.to_csv(
        output_dir / f"MLP_{coordinate_label}_all_predictions.csv", index=False
    )
    metrics.to_csv(
        output_dir / f"MLP_{coordinate_label}_all_metrics.csv", index=False
    )
    for split_type in ["spatial", "random"]:
        predictions[predictions["split_type"] == split_type].to_csv(
            output_dir / f"MLP_{coordinate_label}_{split_type}_predictions.csv",
            index=False,
        )
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
