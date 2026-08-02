# Random Forest Baseline

Random Forest models for the DATA5925 reptile species distribution modelling project.

## Files

- `rf_baseline.py`: trains one Random Forest classifier per species (single-species approach).
- `rf_multi_species.py`: compares single-species vs multi-species (one-hot) vs multi-species (+interactions) Random Forests.
- `split_summary.csv`: per-species, per-class summary of the train/test split.
- `rf_single_vs_multi_results.csv`: per-species log loss comparison of the three strategies.

## Data split

Both scripts use the **team's shared row-level split**, loaded directly from
`Train and Test/train_split.csv` and `Train and Test/test_split.csv`. Do not
re-split the data in this folder — all team models are evaluated on the same
split so results are directly comparable.

Note: this is a **row-level stratified split** — rows from the same plot can
appear in both train and test. This differs from the plot-level split in
`MLP/train_plotlevel.csv` / `MLP/test_plotlevel.csv`, which keeps all 8 species
rows of a plot together to avoid spatial leakage.

## Key design choices

- Uses `easting`/`northing` as spatial predictors (not `long`/`lat`).
- Trains one model per species (single-species approach) in `rf_baseline.py`.
- Uses `class_weight='balanced'` to handle severe class imbalance.
- Evaluates with log loss, AUC-ROC, Brier score, F1, sensitivity, and specificity.

## How to run

From the repository root:

```bash
cd "Random Forest"
python rf_baseline.py
python rf_multi_species.py
```

The scripts expect the Kaggle data (`test.csv`) in
`../predicting-small-reptile-species-distributions-in-nsw/` relative to the
repository root, and the shared split files in `../Train and Test/`.

## Output

- Console: overall and per-species test metrics.
- `submission_rf_baseline.csv`, `submission_rf_single_species.csv`,
  `submission_rf_multi_species.csv`: Kaggle-ready submission files.
- `split_summary.csv`: summary of the split counts.
- `rf_single_vs_multi_results.csv`: per-species strategy comparison.
