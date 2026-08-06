# Shared modelling protocol

This folder provides a shared protocol so that GAM, Random Forest, XGBoost, Logistic Regression, and MLP use the same data splits, predictors, preprocessing rules, evaluation metrics, and prediction format. `protocol_config.json` stores these common settings, and `protocol.py` provides the reusable functions for loading the data, selecting predictors, preprocessing variables, tuning a model on training data, evaluating it on held-out data, and exporting standard results. Each model family can still define its own estimator and hyperparameter grid.

The `gam_example.py` file currently shown in the GitHub `protocol` folder is the GAM rerun example with coordinate predictors switched off. It uses `USE_COORDS = False`, so `easting` and `northing` are excluded from the model inputs but are retained in the prediction output for later spatial diagnostics. The script compares single-species, pooled multi-species, and species-interaction GAMs using both the spatial split and the random split. Hyperparameters are selected through five-fold stratified, plot-grouped cross-validation on the relevant training split only. The selected model is then refitted using the complete training split and evaluated once on its held-out test split.

Place the following four data files in the same directory as `gam_example.py`, `protocol.py`, and `protocol_config.json`:

- `spatial_train.csv`
- `spatial_test.csv`
- `random_train.csv`
- `random_test.csv`

Run the example from that directory:

```bash
python gam_example.py
```

The script uses its own directory as the default data and output directory. Different locations can be supplied when necessary:

```bash
python gam_example.py --repo-root "C:/path/to/split-files" --output-dir "C:/path/to/results"
```

The generated CSV files contain cross-validation results, selected parameters, held-out evaluation metrics, and predictions. Combined prediction and metric files are also produced for the spatial and random comparisons. To use another model family, keep the shared configuration and protocol functions unchanged, replace the GAM estimator and model-specific parameter grid, and export predictions in the same standard format.
