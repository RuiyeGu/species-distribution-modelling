# MLP — Multi-Species Neural Network Models

This folder contains the multilayer perceptron (MLP) part of our reptile species
distribution modelling project. It answers two questions: does modelling several
species together beat modelling them one at a time, and does the way we validate
the model change that answer?

## How to run it

Set up the environment once:

```bash
pip install -r requirements.txt
```

Then run the scripts in this order:

```bash
python3 sdm_protocol.py          # 1. builds the train/test splits and CV folds
python3 mlp_experiments.py       # 2. runs all the MLP models
python3 direction_e.py           # 3. compares random vs spatial validation
python3 class_weight_ablation.py # 4. tests the effect of class weighting
python3 graphs_figures.py        # 5. makes the thesis figures
python3 make_slides_figures.py   # 6. makes the presentation figures
```

Each script prints a summary to the screen and writes its results to the files
described below.

## What each output file means

**Model results**

| File | What it contains |
|------|------------------|
| `mlp_summary.csv` | The headline table: how each model version scored, under each validation method. Start here. |
| `mlp_results.csv` | The full detail — every score for every species (raw, nothing hidden). |
| `mlp_results_reported.csv` | The same as above, but with unreliable rare-species scores blanked out. This is the version used in the thesis tables. |
| `mlp_oof.csv` | The single-vs-multi comparison per species, used to show how the answer flips between validation methods. |
| `submission_mlp.csv` | The predictions submitted to the Kaggle competition. |

**Validation and imbalance analysis**

| File | What it contains |
|------|------------------|
| `direction_e_results.csv` | Why random validation is over-optimistic — the spatial-autocorrelation analysis. |
| `ablation_results.csv` | The class-weighting experiment: what weighting does to accuracy vs. calibration. |
| `calibration_plot.png` | The picture version of the weighting result. |

**Figures**

| Folder | What it contains |
|--------|------------------|
| `figures/` | The charts used in the written thesis. |
| `slides/` | Bigger, simpler versions of the same charts for the presentation. |

**Inputs and setup**

| File | What it contains |
|------|------------------|
| `train_plotlevel.csv`, `test_plotlevel.csv` | The training and test data. |
| `train_rowlevel.csv`, `test_rowlevel.csv` | An alternative split used for comparison with teammates' models. |
| `kaggle_test.csv` | The competition's held-out data (no answers). |
| `protocol_splits.json` | The saved fold assignments, so results are reproducible. |
| `requirements.txt` | The exact package versions used. |
