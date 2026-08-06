"""Run diagnostics for predictions with and without coordinates."""

from pathlib import Path

import pandas as pd

from spatial_diagnostics import (
    compute_gap_tables,
    compute_moran_table,
    load_predictions,
    nearest_training_distances,
)


K_NEIGHBOURS = 8
PERMUTATIONS = 999
SEED = 42


def main():
    root = Path(__file__).resolve().parent

    prediction_pairs = {
        "with_coords": [
            root / "GAM_spatial_predictions.csv",
            root / "GAM_random_predictions.csv",
        ],
        "no_coords": [
            root / "GAM_no_coords_spatial_predictions.csv",
            root / "GAM_no_coords_random_predictions.csv",
        ],
    }

    # Check that all required prediction files exist.
    missing = [
        str(path)
        for paths in prediction_pairs.values()
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing prediction files:\n"
            + "\n".join(missing)
        )

    # Calculate each test plot's distance to its nearest training plot.
    distances = pd.concat(
        [
            nearest_training_distances(root, "spatial"),
            nearest_training_distances(root, "random"),
        ],
        ignore_index=True,
    )

    distance_summary = (
        distances
        .groupby("split_type")["distance_km"]
        .agg(
            n_test_plots="count",
            median_km="median",
            mean_km="mean",
            min_km="min",
            max_km="max",
        )
        .reset_index()
    )

    distances.to_csv(
        root / "nearest_training_distances.csv",
        index=False,
    )

    distance_summary.to_csv(
        root / "distance_summary.csv",
        index=False,
    )

    all_moran = []
    all_gaps = []
    all_correlations = []

    # Analyse predictions with coordinates and without coordinates.
    for coordinate_setting, paths in prediction_pairs.items():
        predictions = load_predictions(paths)

        moran = compute_moran_table(
            predictions,
            K_NEIGHBOURS,
            PERMUTATIONS,
            SEED,
        )

        gaps, correlations = compute_gap_tables(
            predictions,
            moran,
            PERMUTATIONS,
            SEED,
        )

        # Mark which coordinate setting produced each result.
        moran.insert(
            0,
            "coordinate_setting",
            coordinate_setting,
        )

        gaps.insert(
            0,
            "coordinate_setting",
            coordinate_setting,
        )

        correlations.insert(
            0,
            "coordinate_setting",
            coordinate_setting,
        )

        # Save results for this coordinate setting.
        moran.to_csv(
            root / f"{coordinate_setting}_morans_i.csv",
            index=False,
        )

        gaps.to_csv(
            root / f"{coordinate_setting}_optimism_gaps.csv",
            index=False,
        )

        correlations.to_csv(
            root
            / f"{coordinate_setting}_gap_moran_correlations.csv",
            index=False,
        )

        all_moran.append(moran)
        all_gaps.append(gaps)
        all_correlations.append(correlations)

    # Save combined results.
    pd.concat(
        all_moran,
        ignore_index=True,
    ).to_csv(
        root / "all_morans_i.csv",
        index=False,
    )

    pd.concat(
        all_gaps,
        ignore_index=True,
    ).to_csv(
        root / "all_optimism_gaps.csv",
        index=False,
    )

    pd.concat(
        all_correlations,
        ignore_index=True,
    ).to_csv(
        root / "all_gap_moran_correlations.csv",
        index=False,
    )

    print(distance_summary.to_string(index=False))
    print(
        f"Diagnostics complete. "
        f"Results saved to {root}"
    )


if __name__ == "__main__":
    main()