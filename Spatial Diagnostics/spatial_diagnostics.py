"""reusable spatial diagnostics for any model's predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "plot", "species", "model", "split_type", "y_true", "y_prob",
    "easting", "northing",
}


def load_predictions(paths):
    tables = [pd.read_csv(path) for path in paths]
    predictions = pd.concat(tables, ignore_index=True)
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction files are missing columns: {sorted(missing)}")
    if not set(predictions["split_type"]).issubset({"spatial", "random"}):
        raise ValueError("split_type must contain only 'spatial' and 'random'")
    if predictions["y_prob"].isna().any() or not predictions["y_prob"].between(0, 1).all():
        raise ValueError("y_prob must contain finite probabilities in [0, 1]")
    return predictions


def nearest_training_distances(repo_root, split_type):
    repo_root = Path(repo_root)
    split_dir = repo_root
    if not (split_dir / f"{split_type}_train.csv").exists():
        split_dir = repo_root / "Split data"
    train = pd.read_csv(split_dir / f"{split_type}_train.csv")
    test = pd.read_csv(split_dir / f"{split_type}_test.csv")
    train_plots = train.drop_duplicates("plot")[["plot", "easting", "northing"]]
    test_plots = test.drop_duplicates("plot")[["plot", "easting", "northing"]]

    train_xy = train_plots[["easting", "northing"]].to_numpy(float)
    rows = []
    for point in test_plots.itertuples(index=False):
        delta = train_xy - np.array([point.easting, point.northing])
        distance = np.sqrt(np.sum(delta * delta, axis=1))
        nearest_index = int(np.argmin(distance))
        rows.append({
            "split_type": split_type,
            "test_plot": point.plot,
            "nearest_train_plot": train_plots.iloc[nearest_index]["plot"],
            "distance_coordinate_units": float(distance[nearest_index]),
            "distance_km": float(distance[nearest_index] / 1000.0),
        })
    return pd.DataFrame(rows)


def knn_weights(coordinates, k):
    n = len(coordinates)
    if n < 3:
        raise ValueError("Moran's I requires at least three test plots")
    k = min(k, n - 1)
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    weights = np.zeros((n, n), dtype=float)
    weights[np.arange(n)[:, None], neighbours] = 1.0
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def morans_i(values, weights):
    values = np.asarray(values, dtype=float)
    centred = values - values.mean()
    denominator = float(centred @ centred)
    if denominator == 0:
        return np.nan
    return float(len(values) / weights.sum() * ((weights * np.outer(centred, centred)).sum() / denominator))


def moran_permutation_test(values, coordinates, k, permutations, rng):
    weights = knn_weights(np.asarray(coordinates, dtype=float), k)
    observed = morans_i(values, weights)
    if np.isnan(observed):
        return observed, np.nan
    permuted = np.array([morans_i(rng.permutation(values), weights) for _ in range(permutations)])
    p_value = (1 + np.sum(np.abs(permuted) >= abs(observed))) / (permutations + 1)
    return observed, float(p_value)


def compute_moran_table(predictions, k, permutations, seed):
    rng = np.random.default_rng(seed)
    rows = []
    keys = ["model", "species", "split_type"]
    for key, group in predictions.groupby(keys, sort=True):
        if group["plot"].duplicated().any():
            raise ValueError(f"Duplicate plot predictions found for {key}")
        residual = group["y_true"].to_numpy(float) - group["y_prob"].to_numpy(float)
        coordinates = group[["easting", "northing"]].to_numpy(float)
        statistic, p_value = moran_permutation_test(
            residual, coordinates, k, permutations, rng
        )
        rows.append({
            "model": key[0],
            "species": key[1],
            "split_type": key[2],
            "n_test_plots": len(group),
            "k_neighbours": min(k, len(group) - 1),
            "permutations": permutations,
            "morans_i": statistic,
            "permutation_p": p_value,
        })
    return pd.DataFrame(rows)


def binary_auc(y_true, probability):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    positive = probability[y_true == 1]
    negative = probability[y_true == 0]
    if len(positive) == 0 or len(negative) == 0:
        return np.nan
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def rank_values(values):
    return pd.Series(values).rank(method="average").to_numpy(float)


def spearman(values_a, values_b):
    a = rank_values(values_a)
    b = rank_values(values_b)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman_permutation_test(values_a, values_b, permutations, rng):
    observed = spearman(values_a, values_b)
    if np.isnan(observed):
        return observed, np.nan
    permuted = np.array([spearman(values_a, rng.permutation(values_b)) for _ in range(permutations)])
    p_value = (1 + np.sum(np.abs(permuted) >= abs(observed))) / (permutations + 1)
    return observed, float(p_value)


def compute_gap_tables(predictions, moran_table, permutations, seed):
    score_rows = []
    for key, group in predictions.groupby(["model", "species", "split_type"], sort=True):
        score_rows.append({
            "model": key[0],
            "species": key[1],
            "split_type": key[2],
            "auc": binary_auc(group["y_true"], group["y_prob"]),
        })
    scores = pd.DataFrame(score_rows)
    wide = scores.pivot(index=["model", "species"], columns="split_type", values="auc").reset_index()
    if not {"random", "spatial"}.issubset(wide.columns):
        raise ValueError("Both random and spatial predictions are required for optimism gaps")
    wide["optimism_gap_auc"] = wide["random"] - wide["spatial"]
    spatial_moran = moran_table[moran_table["split_type"] == "spatial"][[
        "model", "species", "morans_i"
    ]]
    gaps = wide.merge(spatial_moran, on=["model", "species"], how="left")

    rng = np.random.default_rng(seed)
    correlations = []
    for model, group in gaps.dropna(subset=["optimism_gap_auc", "morans_i"]).groupby("model"):
        rho, p_value = spearman_permutation_test(
            group["optimism_gap_auc"].to_numpy(),
            group["morans_i"].to_numpy(),
            permutations,
            rng,
        )
        correlations.append({
            "model": model,
            "n_species": len(group),
            "spearman_rho": rho,
            "permutation_p": p_value,
            "permutations": permutations,
        })
    return gaps, pd.DataFrame(correlations)


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=script_dir)
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        default=[
            script_dir / "GAM_spatial_predictions.csv",
            script_dir / "GAM_random_predictions.csv",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(args.predictions)
    distances = pd.concat([
        nearest_training_distances(args.repo_root, "spatial"),
        nearest_training_distances(args.repo_root, "random"),
    ], ignore_index=True)
    distance_summary = distances.groupby("split_type")["distance_km"].agg(
        n_test_plots="count", median_km="median", mean_km="mean", min_km="min", max_km="max"
    ).reset_index()
    moran = compute_moran_table(predictions, args.k, args.permutations, args.seed)
    gaps, correlations = compute_gap_tables(predictions, moran, args.permutations, args.seed)

    distances.to_csv(args.output_dir / "nearest_training_distances.csv", index=False)
    distance_summary.to_csv(args.output_dir / "distance_summary.csv", index=False)
    moran.to_csv(args.output_dir / "morans_i.csv", index=False)
    gaps.to_csv(args.output_dir / "optimism_gaps.csv", index=False)
    correlations.to_csv(args.output_dir / "gap_moran_correlations.csv", index=False)
    print(distance_summary.to_string(index=False))
    print(f"Saved Task 4 results to {args.output_dir}")


if __name__ == "__main__":
    main()
