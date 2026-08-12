import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from protocol import (
    evaluate,
    get_predictors,
    load_config,
    load_split,
    run_tuned_experiment,
)


CONFIG = load_config()
TARGET = CONFIG["target"]
SPECIES = CONFIG["species"]
GROUP = CONFIG["group"]
ID = CONFIG["id"]
SEED = CONFIG["random_seed"]
N_SPLITS = CONFIG["tuning"]["n_splits"]
MAX_ITER = 500

# Shared across both grids on purpose: if the single and multi paths tune over
# different alpha ranges the single-vs-multi comparison stops being like-for-like.
# The range extends past 0.1 because with the earlier [1e-4 .. 1e-1] grid the
# single path selected the 0.1 ceiling in 27 of 32 fits (8/8 in both no_coords
# cells), i.e. tuning wanted more regularisation than the grid could offer.
HIDDEN_LAYER_SIZES = [(8,), (16,), (32,), (16, 8)]
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0]
LEARNING_RATE_INIT = [1e-2, 1e-3]

MLP_PARAM_GRID = {
    "model__hidden_layer_sizes": HIDDEN_LAYER_SIZES,
    "model__alpha": ALPHAS,
    "model__learning_rate_init": LEARNING_RATE_INIT,
}

MLP_MULTI_GRID = {
    "classifier__hidden_layer_sizes": HIDDEN_LAYER_SIZES,
    "classifier__alpha": ALPHAS,
    "classifier__learning_rate_init": LEARNING_RATE_INIT,
}


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:  # older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def make_single_mlp():
    return MLPClassifier(
        activation="relu",
        solver="adam",
        max_iter=MAX_ITER,
        random_state=SEED,
    )


class MultiMLPTransformer(BaseEstimator, TransformerMixin):


    def __init__(self, env_features):
        self.env_features = env_features

    def fit(self, X, y=None):
        self.imputer_ = SimpleImputer(strategy="median").fit(X[self.env_features])
        imputed = self.imputer_.transform(X[self.env_features])
        self.scaler_ = StandardScaler().fit(imputed)
        self.species_ = one_hot_encoder().fit(X[[SPECIES]])
        return self

    def transform(self, X):
        environment = self.scaler_.transform(
            self.imputer_.transform(X[self.env_features])
        )
        species = self.species_.transform(X[[SPECIES]])
        return np.hstack([environment, species])


def make_multi_mlp(env_features):
    return Pipeline([
        ("features", MultiMLPTransformer(env_features)),
        ("classifier", MLPClassifier(
            activation="relu",
            solver="adam",
            max_iter=MAX_ITER,
            random_state=SEED,
        )),
    ])


def tune_multi(train, test, split_type, approach, output_dir, use_coords):
    env_features = get_predictors(use_coords)
    model = make_multi_mlp(env_features)

    X_train = train[[SPECIES] + env_features].reset_index(drop=True)
    y_train = train[TARGET].astype(int).reset_index(drop=True)
    groups = train[GROUP].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    cv_rows = []
    candidates = []
    for candidate, params in enumerate(ParameterGrid(MLP_MULTI_GRID), start=1):
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
                "use_coords": use_coords,
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
        final_model.predict_proba(test[[SPECIES] + env_features])[:, 1],
        1e-6,
        1 - 1e-6,
    )

    predictions = pd.DataFrame({
        "plot": test[GROUP].to_numpy(),
        "id": test[ID].to_numpy(),
        "species": test[SPECIES].to_numpy(),
        "model": approach,
        "split_type": split_type,
        "use_coords": use_coords,
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
            "use_coords": use_coords,
            "species": species,
            **evaluate(group["y_true"], group["y_prob"]),
        })
    metrics = pd.DataFrame(metric_rows)
    best_params = pd.DataFrame([{
        "model": approach,
        "split_type": split_type,
        "use_coords": use_coords,
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


from sklearn.model_selection import KFold

MLP_TRUNK_GRID = {
    "classifier__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
    "classifier__alpha": ALPHAS,
    "classifier__learning_rate_init": [1e-2, 1e-3],
}


def to_wide(frame, env_features):
    features = (frame.drop_duplicates(GROUP)
                .set_index(GROUP)[env_features + ["easting", "northing"]])
    features = features.loc[:, ~features.columns.duplicated()]
    targets = frame.pivot(index=GROUP, columns=SPECIES, values=TARGET)
    ids = frame.pivot(index=GROUP, columns=SPECIES, values=ID)
    return features.join(targets, rsuffix="_y"), targets, ids


def make_trunk_mlp(env_features):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", MLPClassifier(activation="relu", solver="adam",
                                     max_iter=MAX_ITER, random_state=SEED)),
    ])


def _trunk_proba(model, X, n_species):
    proba = model.predict_proba(X)
    proba = np.asarray(proba)
    if proba.ndim == 3:  # some versions return a list of per-output arrays
        proba = np.stack([p[:, 1] for p in proba], axis=1)
    return proba[:, :n_species]


def tune_multi_trunk(train, test, split_type, approach, output_dir, use_coords):
    env_features = get_predictors(use_coords)
    species_names = sorted(train[SPECIES].unique())
    n_species = len(species_names)

    wide_train, y_train_df, _ = to_wide(train, env_features)
    wide_test, y_test_df, id_test_df = to_wide(test, env_features)
    X_train = wide_train[env_features].to_numpy(dtype=float)
    Y_train = y_train_df[species_names].to_numpy(dtype=int)
    X_test = wide_test[env_features].to_numpy(dtype=float)

    model = make_trunk_mlp(env_features)
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    cv_rows, candidates = [], []
    for candidate, params in enumerate(ParameterGrid(MLP_TRUNK_GRID), start=1):
        losses = []
        for fold, (fit_idx, val_idx) in enumerate(splitter.split(X_train), start=1):
            fitted = clone(model).set_params(**params)
            fitted.fit(X_train[fit_idx], Y_train[fit_idx])
            proba = _trunk_proba(fitted, X_train[val_idx], n_species)
            metrics = evaluate(Y_train[val_idx].ravel(), proba.ravel())
            losses.append(metrics["log_loss"])
            cv_rows.append({"model": approach, "split_type": split_type,
                            "use_coords": use_coords, "candidate": candidate,
                            "fold": fold, **params, **metrics})
        candidates.append({"params": params,
                           "mean_cv_log_loss": float(np.mean(losses)),
                           "std_cv_log_loss": float(np.std(losses))})

    best = min(candidates, key=lambda row: row["mean_cv_log_loss"])
    final_model = clone(model).set_params(**best["params"])
    final_model.fit(X_train, Y_train)
    proba = np.clip(_trunk_proba(final_model, X_test, n_species), 1e-6, 1 - 1e-6)

    records = []
    for row, plot in enumerate(wide_test.index):
        for col, species in enumerate(species_names):
            records.append({
                "plot": plot,
                "id": id_test_df.loc[plot, species],
                "species": species,
                "model": approach,
                "split_type": split_type,
                "use_coords": use_coords,
                "y_true": int(y_test_df.loc[plot, species]),
                "y_prob": float(proba[row, col]),
                "easting": wide_test.loc[plot, "easting"],
                "northing": wide_test.loc[plot, "northing"],
            })
    predictions = pd.DataFrame(records)

    metric_rows = [{"model": approach, "split_type": split_type,
                    "use_coords": use_coords, "species": species,
                    **evaluate(group["y_true"], group["y_prob"])}
                   for species, group in predictions.groupby("species")]
    metrics = pd.DataFrame(metric_rows)
    best_params = pd.DataFrame([{"model": approach, "split_type": split_type,
                                 "use_coords": use_coords, **best["params"],
                                 "mean_cv_log_loss": best["mean_cv_log_loss"],
                                 "std_cv_log_loss": best["std_cv_log_loss"]}])

    prefix = f"{approach}_{split_type}"
    predictions.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    metrics.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output_dir / f"{prefix}_cv_results.csv", index=False)
    best_params.to_csv(output_dir / f"{prefix}_best_params.csv", index=False)
    return predictions, metrics, pd.DataFrame(cv_rows), best_params


def run_for_coords(use_coords, repo_root, output_dir):
    coordinate_label = "with_coords" if use_coords else "no_coords"
    all_predictions = []
    all_metrics = []
    for split_type in ["spatial", "random"]:
        train, test = load_split(repo_root, split_type)

        single_predictions, single_metrics, _, _ = run_tuned_experiment(
            model_name=f"MLP_single_{coordinate_label}",
            estimator=make_single_mlp(),
            param_grid=MLP_PARAM_GRID,
            split_type=split_type,
            repo_root=repo_root,
            output_dir=output_dir,
            use_coords=use_coords,
        )
        all_predictions.append(single_predictions)
        all_metrics.append(single_metrics)

        multi_predictions, multi_metrics, _, _ = tune_multi(
            train, test, split_type,
            approach=f"MLP_multi_{coordinate_label}",
            output_dir=output_dir,
            use_coords=use_coords,
        )
        all_predictions.append(multi_predictions)
        all_metrics.append(multi_metrics)

        trunk_predictions, trunk_metrics, _, _ = tune_multi_trunk(
            train, test, split_type,
            approach=f"MLP_multi_trunk_{coordinate_label}",
            output_dir=output_dir,
            use_coords=use_coords,
        )
        all_predictions.append(trunk_predictions)
        all_metrics.append(trunk_metrics)

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
    return metrics


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]  # protocol/MLP -> repo root
    default_data = repo_root / "Split data" / "dataset"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=default_data,
        help="Directory holding {spatial,random}_{train,test}.csv "
             "(default: 'Split data/dataset').",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "Output")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for use_coords in (True, False):
        metrics = run_for_coords(use_coords, args.repo_root, output_dir)
        overall = (
            metrics.groupby("model")["log_loss"].mean().round(4).to_dict()
        )
        summary.append((use_coords, overall))

    print("Saved MLP protocol outputs to", output_dir)
    for use_coords, overall in summary:
        label = "with_coords" if use_coords else "no_coords"
        print(f"\n[{label}] mean per-species log loss by model group:")
        for model_name, value in overall.items():
            print(f"  {model_name:42s} {value}")


if __name__ == "__main__":
    main()