# Data Split

This code creates fixed **spatial** and **random** 80/20 train-test splits from
`train.csv`. The original data contain 641 plots and eight species records for
each plot. All rows belonging to the same plot are kept together to prevent
plot leakage between training and test data.

## Spatial split

The plots are sorted by `northing`, and every continuous 20% band is checked.
A band is valid only when the following four data-rich species have presences
in both the training and test sets:

- `Calyptotis scutirostrum`
- `Coeranoscincus reticulatus`
- `Ophioscincus truncatus`
- `Cacophis kreftii`

Among the valid bands, the code selects the one with the largest median
distance from each test plot to its nearest training plot. This provides a
geographically separated test region.

## Random split

The random split uses the same number of test plots as the spatial split and a
fixed random seed (`42`). Complete plots are randomly assigned, so the same
plot cannot appear in both sets.

## Outputs

The code saves:

- `spatial_train.csv` and `spatial_test.csv`
- `random_train.csv` and `random_test.csv`
- presence summaries for both splits
- nearest-training-distance files for both splits
- `data_splits.json`, which records the split settings and plot IDs

The script also checks that no plots overlap and no plots or rows are lost.
The four sparse species are not required to have presences in both test sets,
so some evaluation metrics may be `NaN` when a test set contains no positive
observations for one of these species.
