# Random Forest Baseline

Random Forest baseline for the DATA5925 reptile species distribution modelling project.

## Files

- `rf_baseline.py`: trains one Random Forest classifier per species using a **plot-level** train/validation split.
- `split_summary.csv`: per-species, per-class summary of the train/validation split.
- `train_split.csv` / `test_split.csv`: the actual plot-level split used for validation.

## Key design choices

- Uses `easting`/`northing` as spatial predictors (not `long`/`lat`).
- Splits by `plot` using `GroupShuffleSplit` to avoid spatial leakage between train and validation.
- Trains one model per species (single-species approach).
- Uses `class_weight='balanced'` to handle severe class imbalance.
- Evaluates with log loss, AUC-ROC, Brier score, F1, sensitivity, and specificity.

## How to run

From the repository root:

```bash
cd "Random Forest"
python rf_baseline.py
```

## Output

- Console: overall and per-species validation metrics.
- `submission_rf_baseline.csv`: Kaggle-ready submission file (written to the project `outputs/` folder when run from the original location; adjust path if needed).
- `split_summary.csv`: summary of the split counts.

## Note on train/test splitting

This script uses a **plot-level split**, meaning all 8 species observations from the same plot stay together in either the training or validation set. This follows Nickson Ning's guidance in the Teams channel to avoid spatial leakage. It differs from the row-level stratified split in `Train and Test/`; the team should decide on a consistent splitting strategy for final model comparison.
