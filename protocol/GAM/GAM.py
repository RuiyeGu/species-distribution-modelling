"""Task 6: tuned GAM comparisons with easting/northing excluded."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from protocol import (
    evaluate,
    get_predictors,
    load_config,
    load_split,
    run_tuned_experiment,
)


CONFIG = load_config()
# Match the MLP convention: change this one line to switch coordinate inputs.
USE_COORDS = False
ENV_FEATURES = get_predictors(use_coords=USE_COORDS)
TARGET = CONFIG["target"]
SPECIES = CONFIG["species"]
GROUP = CONFIG["group"]
SEED = CONFIG["random_seed"]
N_SPLITS = CONFIG["tuning"]["n_splits"]

GAM_PARAM_GRID = {
    "model__splines__n_knots": [4, 6, 8],
    "model__classifier__C": [0.1, 0.5],
    "model__classifier__class_weight": [None, "balanced"],
}


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


class MultiGAMTransformer(BaseEstimator, TransformerMixin):
    """Training-fitted environmental splines plus Species effects/interactions."""

    def __init__(self, n_knots=6, interactions=False):
        self.n_knots = n_knots
        self.interactions = interactions

    def fit(self, X, y=None):
        self.imputer_ = SimpleImputer(strategy="median").fit(X[ENV_FEATURES])
        imputed = self.imputer_.transform(X[ENV_FEATURES])
        self.scaler_ = StandardScaler().fit(imputed)
        scaled = self.scaler_.transform(imputed)
        self.splines_ = SplineTransformer(
            n_knots=self.n_knots, degree=3, include_bias=False
        ).fit(scaled)
        self.species_ = one_hot_encoder().fit(X[[SPECIES]])
        return self

    def transform(self, X):
        imputed = self.imputer_.transform(X[ENV_FEATURES])
        smooth = self.splines_.transform(self.scaler_.transform(imputed))
        species = self.species_.transform(X[[SPECIES]])
        parts = [smooth, species]
        if self.interactions:
            interaction = (
                species[:, :, np.newaxis] * smooth[:, np.newaxis, :]
            ).reshape(len(X), -1)
            parts.append(interaction)
        return np.hstack(parts)


def make_single_gam():
    return Pipeline([
        ("splines", SplineTransformer(degree=3, include_bias=False)),
        ("classifier", LogisticRegression(
            solver="lbfgs", max_iter=5000, random_state=SEED
        )),
    ])


def make_multi_gam(interactions=False):
    return Pipeline([
        ("features", MultiGAMTransformer(interactions=interactions)),
        ("classifier", LogisticRegression(
            solver="lbfgs", max_iter=5000, random_state=SEED
        )),
    ])


def tune_multi(train, test, split_type, approach, interactions, output_dir):
    model = make_multi_gam(interactions)
    grid = {
        "features__n_knots": [4, 6, 8],
        "classifier__C": [0.1, 0.5],
        "classifier__class_weight": [None, "balanced"],
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
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    all_metrics = []
    coordinate_label = "with_coords" if USE_COORDS else "no_coords"
    for split_type in ["spatial", "random"]:
        train, test = load_split(args.repo_root, split_type)
        single_predictions, single_metrics, _, _ = run_tuned_experiment(
            model_name=f"GAM_single_{coordinate_label}",
            estimator=make_single_gam(),
            param_grid=GAM_PARAM_GRID,
            split_type=split_type,
            repo_root=args.repo_root,
            output_dir=output_dir,
            use_coords=USE_COORDS,
        )
        all_predictions.append(single_predictions)
        all_metrics.append(single_metrics)

        for approach, interactions in [
            (f"GAM_multi_{coordinate_label}", False),
            (f"GAM_multi_interactions_{coordinate_label}", True),
        ]:
            predictions, metrics, _, _ = tune_multi(
                train, test, split_type, approach, interactions, output_dir
            )
            all_predictions.append(predictions)
            all_metrics.append(metrics)

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions.to_csv(
        output_dir / f"GAM_{coordinate_label}_all_predictions.csv", index=False
    )
    metrics.to_csv(
        output_dir / f"GAM_{coordinate_label}_all_metrics.csv", index=False
    )
    for split_type in ["spatial", "random"]:
        predictions[predictions["split_type"] == split_type].to_csv(
            output_dir / f"GAM_{coordinate_label}_{split_type}_predictions.csv",
            index=False,
        )
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
