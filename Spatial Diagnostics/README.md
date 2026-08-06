# Spatial diagnostics

This script consumes standard prediction CSV files produced by the Task 3
protocol. Both spatial and random predictions for the same model are required.

File names and locations may differ between model families. Pass the actual
prediction file paths to `--predictions`.

Prediction files must contain:

- `plot`
- `species`
- `model`
- `split_type`
- `y_true`
- `y_prob`
- `easting`
- `northing`

Run from the repository root:

```bash
python /spatial_diagnostics.py \
  --repo-root . \
  --predictions <spatial_predictions.csv> <random_predictions.csv> \
  --output-dir <output_directory>