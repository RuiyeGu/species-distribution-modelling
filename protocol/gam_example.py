"""Example showing how GAM plugs into the shared Task 3 protocol."""

import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from protocol import run_experiment


gam = Pipeline([
    ("splines", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
    ("classifier", LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )),
])


def main():
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or repo_root / "GAM" / "Output" / "task3"

    for split_type in ["spatial", "random"]:
        _, metrics = run_experiment(
            model_name="GAM",
            estimator=gam,
            split_type=split_type,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        print(f"\n{split_type} results")
        print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
