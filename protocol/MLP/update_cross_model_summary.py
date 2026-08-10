"""Re-source the MLP row of the cross-model tuned summary from protocol/MLP.

The other four rows of tuned_summary_across_models.csv were produced by
"Single species vs Multi Species/All model/02_hyperparameter_tuning.ipynb".
This script leaves those rows untouched and replaces only the MLP row with
numbers computed from protocol/MLP/Output/, i.e. from the shared protocol.

"overall_log_loss" is defined exactly as the notebook defines it: a single
pooled log_loss over all test rows at once (clip 1e-6, labels=[0, 1]) -- not a
mean of per-species log-losses.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

CLIP = 1e-6


def pooled_log_loss(frame):
    probability = np.clip(frame["y_prob"].to_numpy(float), CLIP, 1 - CLIP)
    return float(log_loss(frame["y_true"].astype(int), probability, labels=[0, 1]))


def mlp_matrix(output_dir):
    """Pooled single/multi log-loss for every coordinate setting x split."""
    rows = []
    for coords in ["with_coords", "no_coords"]:
        for split in ["random", "spatial"]:
            entry = {"use_coords": coords, "split_type": split}
            for approach in ["single", "multi"]:
                path = output_dir / f"MLP_{approach}_{coords}_{split}_predictions.csv"
                entry[f"overall_log_loss_{approach}_tuned"] = pooled_log_loss(
                    pd.read_csv(path)
                )
            entry["multi_wins_after_tuning"] = (
                entry["overall_log_loss_multi_tuned"]
                < entry["overall_log_loss_single_tuned"]
            )
            rows.append(entry)
    return pd.DataFrame(rows)


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=script_dir / "Output")
    parser.add_argument(
        "--summary",
        type=Path,
        default=repo_root
        / "Single species vs Multi Species"
        / "All model"
        / "outputs"
        / "tuned_summary_across_models.csv",
    )
    # The notebook trained on all eight predictors (coordinates included) and a
    # single non-spatial hold-out, so the random/with_coords cell is the closest
    # protocol equivalent for the shared table.
    parser.add_argument("--use-coords", default="with_coords")
    parser.add_argument("--split-type", default="random")
    args = parser.parse_args()

    matrix = mlp_matrix(args.output_dir)
    # Kept beside the script, not in Output/, so Output/ keeps exactly the same
    # file set as the other families' Output/ folders.
    matrix.to_csv(script_dir / "MLP_overall_log_loss_matrix.csv", index=False)
    print("MLP pooled overall log-loss, all protocol cells:")
    print(matrix.to_string(index=False))

    chosen = matrix[
        (matrix["use_coords"] == args.use_coords)
        & (matrix["split_type"] == args.split_type)
    ].iloc[0]

    # Rewrite the MLP line only, as text. Round-tripping the whole table through
    # pandas re-formats the other families' floats (last-digit changes that are
    # numerically identical but show up as diff noise), so leave their bytes alone.
    lines = args.summary.read_text(encoding="utf-8").splitlines(keepends=True)
    replacement = (
        f"MLP,{float(chosen['overall_log_loss_single_tuned'])!r},"
        f"{float(chosen['overall_log_loss_multi_tuned'])!r},"
        f"{bool(chosen['multi_wins_after_tuning'])}"
    )
    before = None
    for index, line in enumerate(lines):
        if line.split(",", 1)[0] == "MLP":
            before = line.rstrip("\r\n")
            ending = line[len(before):]
            lines[index] = replacement + ending
            break
    if before is None:
        raise ValueError("No MLP row found in the summary table")
    args.summary.write_text("".join(lines), encoding="utf-8")

    print(f"\nMLP row before (notebook-sourced):\n  {before}")
    print(f"MLP row after  (protocol-sourced, {args.use_coords}/{args.split_type}):")
    print(f"  {replacement}")
    print(f"\nUpdated {args.summary}")


if __name__ == "__main__":
    main()
