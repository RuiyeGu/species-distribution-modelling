"""Example showing how GAM plugs into the shared Task 3 protocol."""

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from protocol import run_experiment


# After copying task3_protocol into the repository root, this is its parent.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "GAM" / "Output" 

# GAM-specific component. Shared preprocessing and metrics remain in protocol.py.
gam = Pipeline([
    ("splines", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
    ("classifier", LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )),
])

for split_type in ["spatial", "random"]:
    _, metrics = run_experiment(
        model_name="GAM",
        estimator=gam,
        split_type=split_type,
        repo_root=REPO_ROOT,
        output_dir=OUTPUT_DIR,
    )
    print(f"\n{split_type} results")
    print(metrics.round(4).to_string(index=False))
