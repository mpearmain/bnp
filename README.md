# Kaggle BNP Paribas challenge



All input files and created datasets are stored in `/input` but ignored from git.

All metafeature datasets are stored in `metafeatures` and a seperate gDrive account is active to share these (rather than run and reproduce) and ignored from git.

All submissions are stored in `submissions` but ignored from git.

## Python Dependencies ##

* numpy (<1.10.0)
* scipy (<0.17.0)
* scikit-learn (<0.16.1)
* bayesian-optimization (`pip install git+https://github.com/fmfn/BayesianOptimization.git`)

## Strategy 
In this challenge, BNP Paribas Cardif is providing an anonymized database with two categories of
claims:

1. claims for which approval could be accelerated leading to faster payments.
2. claims for which additional information is required before approval.

This means we are dealing with a binary classification task, and specifically we are measured on 
the logloss metric.

In order to seperate work amongst team members it is important to describe the high level framework
we intend on using (details of specifics can be found subsequent sections), so we are able to
optimize each stage.

1.  Dataset and Feature generation - We aim to create multiple datasets that are diverse in nature
2.  Stacking of level 1 models - For different datasets we create metafeatures based on a variety of
models (XGboost, Extratrees, Factorization Machines, etc.)
3.  Feature selection from level 1 stacked meta features, from the potential 100s of new features
created we must eliminate features for second level stacking and ensembling. 
4. Final blending of level 2 stacked features (based on classic train / validation)
5. Submission.


## Data

### Dataset building

The datasets are generated using two scripts: `./R/build_datasets.R` and `./python/build_datasets.py`. Each dataset is described in a section below - in most cases the naming convention of files is based on those, so e.g. MP1 is stored as `./input/{xtrain,xtest}_MP1.csv`.

In the remainder of this section brackets near the name indicate which script was used to generate a particular dataset.

#### buildMP1 (R)
* count missing values per row
* replace all NA with -1
* map all characters to integers - **this means the MP1 dataset makes sense as input only for tree-based models** 

#### buildKB1 (R)
* count missing values per row
* replace all NA with -1
* addition of quadratic factors (all pairwise combinations of categorical variables)
* map all factors to integers - **this means the KB1 dataset makes sense as input only for tree-based models** 

#### buildKB2 (R)
* count missing values per row
* replace all NA with -1
* addition of quadratic factors (all pairwise combinations of categorical variables)
* addition of cubic factors (all three-way combinations of categorical variables)
* map all factors to integers - **this means the KB2 dataset makes sense as input only for tree-based models** 

#### buildKB3 (R)
* count missing values per row
* replace all NA with -1
* addition of quadratic factors (all pairwise combinations of categorical variables)
* all factors mapped to response rates

#### buildKB4 (R)
* count missing values per row
* replace all NA with -1
* addition of quadratic factors (all pairwise combinations of categorical variables)
* addition of cubic factors (all three-way combinations of categorical variables)
* all factors mapped to response rates via (cross-validated) linear mixed-effects models using the *lmer* package in R

#### buildKB5/buildKB6 (R)



#### KB15


#### KB16
* KB6099 as basis
* SVD (via `sklearn.decomposition.TruncatedSVD`) with `n_components` as function argument

## Running

1. Run in `dataset_creation` run `data_preperation.R`
2. Run all `build_meta_XX.py` in the python subdir
3. We have now produced many metafile that need to be joined into a single dataset
  * `build_linear_combo_selection.R` is an R script to merge all metafiles and remove any 
  linear combinations from the dataset.
  * `build_2ndlLvl_selection.py` takes the output form the above and then build more features by
  ranking the top N results and taking interactions of these vars.
4. TODO: PRODUCE SECOND LEVEL MODELS -> NN / XGB / RF / ET (Only best models)
5. Final stage is to blend the above models weights. (python L-BFGS-L or other optim methos in scipy - mpearmian to produce.)


We follow the convention adopted in Kaggle scripts, so R scripts should be executed from within the R subfolder (relative paths are given as `../submissions` etc) 
