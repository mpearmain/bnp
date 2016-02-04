# bnp
Kaggle BNP Paribus challenge

## Data
All input files and created datasets are stored in `/input` but ignored from git.
All metafeature datasets are stored in `metafeatures` and a seperate gDrive account is active to share these (rather than run and reproduce) and ignored from git.
All submissions are stored in `submissions` but ignored from git.

## Running

1. Run in `dataset_creation` run `data_preperation.R
2. Run all `build_meta_XX.py` in the python subdir
3. Run build_ensemble

To avoid local running problems all files should be run from the top level bnp dir to avoid errors. i.e `./`
