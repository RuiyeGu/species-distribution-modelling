"""
Compare single-species, pooled multi-species, and interactive GAMs.

The implementation reuses gam_baseline.py so all approaches have identical
splines, 5-fold plot-level CV, hyperparameter candidates, and test metrics.
Kaggle prediction is separate and disabled by default.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gam_baseline import (
    ENV_FEATURES,
    MODEL_FEATURES,
    SPECIES,
    TARGET,
    ROOT,
    build_xy,
    clip_probability,
    evaluate_binary,
    make_gam,
    make_per_species_table,
    read_split_data,
    run_multi_species,
    run_single_species,
    selected_params_table,
)

RUN_KAGGLE = False
KAGGLE_APPROACH = "multi_interactions"


# ---------------------------------------------------------------------------
# 1. Comparison tables
# ---------------------------------------------------------------------------

def build_comparison_table(train_df, test_df, predictions):
    tables = []
    for approach, probability in predictions.items():
        table = make_per_species_table(test_df, probability, approach)
        train_summary = (
            train_df.groupby(SPECIES)[TARGET]
            .agg(n_train="size", positives_train="sum", prevalence_train="mean")
            .reset_index()
            .rename(columns={SPECIES: "species"})
        )
        tables.append(table.merge(train_summary, on="species", how="left"))

    long_table = pd.concat(tables, ignore_index=True)
    metric_columns = [
        "log_loss",
        "auc",
        "brier",
        "f1",
        "sensitivity",
        "specificity",
    ]
    base_columns = [
        "species",
        "n_train",
        "n",
        "positives_train",
        "positives",
        "prevalence_train",
        "prevalence",
    ]
    wide = long_table.pivot(index="species", columns="approach", values=metric_columns)
    wide.columns = [f"{metric}_{approach}" for metric, approach in wide.columns]
    species_info = long_table[base_columns].drop_duplicates("species").set_index("species")
    comparison = species_info.join(wide).reset_index()

    loss_columns = {
        "single_species": "log_loss_single_species",
        "multi_species": "log_loss_multi_species",
        "multi_interactions": "log_loss_multi_interactions",
    }
    comparison["best_approach"] = comparison.apply(
        lambda row: min(loss_columns, key=lambda key: row[loss_columns[key]]),
        axis=1,
    )
    return comparison, long_table


def build_overall_summary(test_df, predictions):
    rows = []
    for approach, probability in predictions.items():
        rows.append({
            "approach": approach,
            **evaluate_binary(test_df[TARGET], probability),
        })
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Figure matching the shared single-vs-multi analysis
# ---------------------------------------------------------------------------

def plot_log_loss_comparison(long_table):
    approaches = ["single_species", "multi_species", "multi_interactions"]
    labels = ["Single-species", "Multi-species", "Multi + interactions"]
    colors = ["#2b7bb9", "#f28e2b", "#2ca02c"]
    species = sorted(long_table["species"].unique())
    x = np.arange(len(species))
    width = 0.24

    fig, ax = plt.subplots(figsize=(16, 7.5))
    for offset, approach, label, color in zip(
        [-width, 0, width],
        approaches,
        labels,
        colors,
    ):
        values = (
            long_table[long_table["approach"] == approach]
            .set_index("species")
            .reindex(species)["log_loss"]
        )
        ax.bar(x + offset, values, width, label=label, color=color)

    ax.set_title("GAM: per-species log loss by approach", fontsize=18, pad=14)
    ax.set_ylabel("Log loss (lower is better)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=38, ha="right", fontsize=11)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "GAM_comparison.png", dpi=200, facecolor="white")
    plt.close(fig)


def plot_metric_heatmap(long_table):
    metrics = ["log_loss", "auc", "brier", "f1", "sensitivity", "specificity"]
    approach_labels = {
        "single_species": "Single",
        "multi_species": "Multi",
        "multi_interactions": "Multi + int.",
    }
    rows = []
    row_labels = []
    for approach in approach_labels:
        for species in sorted(long_table["species"].unique()):
            row = long_table[
                (long_table["approach"] == approach)
                & (long_table["species"] == species)
            ].iloc[0]
            rows.append([row[metric] for metric in metrics])
            row_labels.append(f"{approach_labels[approach]} | {species}")

    values = np.asarray(rows, dtype=float)
    normalized = np.empty_like(values)
    for column in range(values.shape[1]):
        current = values[:, column]
        low, high = np.nanmin(current), np.nanmax(current)
        normalized[:, column] = (current - low) / (high - low) if high > low else 0.5

    fig, ax = plt.subplots(figsize=(12, 13))
    image = ax.imshow(normalized, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(["Log loss", "AUC", "Brier", "F1", "Sensitivity", "Specificity"])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("GAM test metrics by species and approach", fontsize=16, pad=12)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text = "NA" if np.isnan(value) else f"{value:.3f}"
            color = "white" if normalized[row, column] > 0.58 else "black"
            ax.text(column, row, text, ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Within-metric relative value")
    fig.tight_layout()
    fig.savefig(ROOT / "GAM_metrics_heatmap.png", dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Optional Kaggle submission
# ---------------------------------------------------------------------------

def create_kaggle_submission(best_params, approach=KAGGLE_APPROACH):
    kaggle_test = pd.read_csv(ROOT / "test.csv")
    sample_submission = pd.read_csv(ROOT / "sample_submission.csv")
    train_df, test_df = read_split_data()
    full_train = pd.concat([train_df, test_df], ignore_index=True)
    probability = np.zeros(len(kaggle_test), dtype=float)

    if approach == "single_species":
        for species in sorted(full_train[SPECIES].unique()):
            train_sp = full_train[full_train[SPECIES] == species].reset_index(drop=True)
            test_mask = kaggle_test[SPECIES] == species
            X_train, y_train, _ = build_xy(train_sp)
            X_kaggle = kaggle_test.loc[test_mask, MODEL_FEATURES].reset_index(drop=True)
            model = make_gam(best_params[approach][species])
            model.fit(X_train, y_train)
            probability[test_mask.to_numpy()] = clip_probability(
                model.predict_proba(X_kaggle)[:, 1]
            )
    else:
        interactions = approach == "multi_interactions"
        X_train, y_train, _ = build_xy(full_train)
        X_kaggle = kaggle_test[MODEL_FEATURES].reset_index(drop=True)
        model = make_gam(
            best_params[approach],
            include_species=True,
            interactions=interactions,
        )
        model.fit(X_train, y_train)
        probability = clip_probability(model.predict_proba(X_kaggle)[:, 1])

    submission = sample_submission.copy()
    submission["pred"] = probability
    output_path = ROOT / f"submission_gam_{approach}.csv"
    submission.to_csv(output_path, index=False)
    print("Saved Kaggle submission:", output_path)
    return submission


# ---------------------------------------------------------------------------
# 4. Main workflow
# ---------------------------------------------------------------------------

def main():
    train_df, test_df = read_split_data()
    print("GAM single-species vs multi-species comparison")
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"All predictors use cubic splines: {ENV_FEATURES}")

    single = run_single_species(train_df, test_df)
    multi = run_multi_species(train_df, test_df, interactions=False)
    interactive = run_multi_species(train_df, test_df, interactions=True)

    predictions = {
        "single_species": single["probability"],
        "multi_species": multi["probability"],
        "multi_interactions": interactive["probability"],
    }
    comparison, long_table = build_comparison_table(train_df, test_df, predictions)
    summary = build_overall_summary(test_df, predictions)

    cv_results = pd.concat([
        single["cv_results"],
        multi["cv_results"],
        interactive["cv_results"],
    ], ignore_index=True)
    selected = pd.concat([
        selected_params_table(single["params"], "single_species"),
        selected_params_table(multi["params"], "multi_species"),
        selected_params_table(interactive["params"], "multi_interactions"),
    ], ignore_index=True)

    comparison.to_csv(ROOT / "GAM_per_species.csv", index=False)
    long_table.to_csv(ROOT / "GAM_full_metrics_per_species.csv", index=False)
    summary.to_csv(ROOT / "GAM_summary.csv", index=False)
    cv_results.to_csv(ROOT / "GAM_cv_results.csv", index=False)
    selected.to_csv(ROOT / "GAM_selected_params.csv", index=False)
    plot_log_loss_comparison(long_table)
    plot_metric_heatmap(long_table)

    print("\nOverall held-out test performance")
    print(summary.round(4).to_string(index=False))
    print("\nPer-species comparison")
    display_columns = [
        "species",
        "log_loss_single_species",
        "log_loss_multi_species",
        "log_loss_multi_interactions",
        "best_approach",
    ]
    print(comparison[display_columns].round(4).to_string(index=False))
    print("\nSaved GAM CSV and PNG outputs in:", ROOT)

    best_params = {
        "single_species": single["params"],
        "multi_species": multi["params"],
        "multi_interactions": interactive["params"],
    }
    if RUN_KAGGLE:
        create_kaggle_submission(best_params, KAGGLE_APPROACH)
    return {
        "best_params": best_params,
        "summary": summary,
        "per_species": comparison,
    }


if __name__ == "__main__":
    RESULTS = main()

    # One-line alternative after training:
    # create_kaggle_submission(RESULTS["best_params"], "multi_interactions")
