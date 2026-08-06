#protocol

The protocol fixes split files, eight predictors, train-only
median imputation and standardisation, threshold, metrics, and output schema.
Each contributor only supplies a scikit-learn-compatible estimator.

Run the GAM example from the repository root:

```bash
python protocol/gam_example.py
```

To add another model, import `run_experiment` and pass an estimator that
implements `fit()` and `predict_proba()`.
