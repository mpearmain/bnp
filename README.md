# Kaggle BNP Paribas challenge

## Data
All input files and created datasets are stored in `/input` but ignored from git.

All metafeature datasets are stored in `metafeatures` and a seperate gDrive account is active to share these (rather than run and reproduce) and ignored from git.

All submissions are stored in `submissions` but ignored from git.

## Strategy 
In this challenge, BNP Paribas Cardif is providing an anonymized database with two categories of
claims:

1. claims for which approval could be accelerated leading to faster payments.
2. claims for which additional information is required before approval.

This means we are dealing with a binary classification task, and specifically we are measured on 
the logloss metric.

In order to seperate work amongst team members it is important to describe the high level framework
we intend on using, (details of specifics can be found subsequent sections) so we are able to
optimize each stage.

1.  Dataset and Feature generation - We aim to create multiple datasets that are diverse in nature
2.  Stacking of level 1 models - For different datasets we create metafeatures based on a variety of
models (XGboost, Extratrees, Factorization Machines, etc.)
3.  Feature selection from level 1 stacked meta features, from the potential 1,000s of new features
created we must eliminate features for second level stacking and ensembling. 
4. Final blending of level 2 stacked features (based on classic train / validation)
5. Submission.

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
