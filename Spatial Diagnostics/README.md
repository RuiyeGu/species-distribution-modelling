# Spatial diagnostics

This folder contains reusable spatial-diagnostic code for analysing held-out predictions. It does not train or rerun a model. `spatial_diagnostics.py` can analyse predictions from any model family, while `run_diagnostics.py` is a convenience runner that compares the GAM results produced with and without coordinate predictors.

The diagnostics calculate each test plot's distance to its nearest training plot, Moran's I for prediction residuals using a permutation test, the AUC optimism gap between random and spatial evaluation, and the Spearman correlation between each species' optimism gap and its spatial residual clustering. Residuals are calculated as `y_true - y_prob`, and the optimism gap is calculated as random-test AUC minus spatial-test AUC. The default settings are 8 nearest neighbours, 999 permutations, and random seed 42.

Prediction CSV files must contain the following columns:

- `plot`
- `species`
- `model`
- `split_type`
- `y_true`
- `y_prob`
- `easting`
- `northing`

The directory supplied as `--repo-root` must contain `spatial_train.csv`, `spatial_test.csv`, `random_train.csv`, and `random_test.csv`. The split files may also be located inside a `Split data` subdirectory. Prediction filenames do not need to follow a fixed naming convention when they are passed explicitly to `--predictions`, but both spatial and random predictions for the same model are required.

To analyse one model or one coordinate setting, run:

```bash
python spatial_diagnostics.py \
  --repo-root <split_csv_directory> \
  --predictions <spatial_predictions.csv> <random_predictions.csv> \
  --output-dir <output_directory>
```

If the split files, prediction files, and scripts are all in the same directory and the prediction files use the default names `GAM_spatial_predictions.csv` and `GAM_random_predictions.csv`, the shorter command can be used:

```bash
python spatial_diagnostics.py
```

This produces:

- `nearest_training_distances.csv`
- `distance_summary.csv`
- `morans_i.csv`
- `optimism_gaps.csv`
- `gap_moran_correlations.csv`

The `run_diagnostics.py` wrapper expects these four GAM prediction files in the same directory as the scripts:

- `GAM_with_coords_spatial_predictions.csv`
- `GAM_with_coords_random_predictions.csv`
- `GAM_no_coords_spatial_predictions.csv`
- `GAM_no_coords_random_predictions.csv`

Run the comparison with:

```bash
python run_diagnostics.py
```

It writes separate Moran's I, optimism-gap, and gap-correlation files for `with_coords` and `no_coords`, as well as combined files named `all_morans_i.csv`, `all_optimism_gaps.csv`, and `all_gap_moran_correlations.csv`. It also writes `nearest_training_distances.csv` and `distance_summary.csv`. The gap-correlation result should be treated as exploratory when only a small number of species is available.
