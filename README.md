# bnp
Kaggle BNP Paribas challenge

## Data
All input files and created datasets are stored in `/input` but ignored from git.

All metafeature datasets are stored in `metafeatures` and a seperate gDrive account is active to share these (rather than run and reproduce) and ignored from git.

All submissions are stored in `submissions` but ignored from git.

### Dataset building

All datasets are generated using the script `./R/build_datasets.R`. Each dataset is described in a sectiob below - the naming convention of files is based on those, so e.g. MP1 is stored as `./input/{xtrain,xtest}_MP1.csv`.

#### MP1
* clone of MP1 from homesite
* count missing values per row
* replace all NA with -1
* map all characters to integers - this means it makes sense as input only for tree-based models

#### KB1
* core transformations from MP1

## Running

1. Run in `dataset_creation` run `data_preperation.R`
2. Run all `build_meta_XX.py` in the python subdir
3. Run build_ensemble

To avoid local running problems all files should be run from the top level bnp dir to avoid errors. i.e `./`
