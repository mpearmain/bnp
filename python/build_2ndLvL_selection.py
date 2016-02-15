import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from python.bortua2 import BorutaPy2

# We need to import all the meta based predictions from ./metafeatures and
# combine these into a single pandas dataframe.
# Once this has been done we are then able to test different feature selection
# methods, i.e bortua or linear combination reductions for the second level
# metas.

# settings
projPath = './'
dataset_version = "ensemble_base"
todate = datetime.datetime.now().strftime("%Y%m%d")
no_bags = 1

## data
# read the training and test sets
xtrain = pd.read_csv(projPath + 'input/xvalid_'+ dataset_version + '.csv')
id_train = xtrain.ID
ytrain = xtrain.target
xtrain.drop('ID', axis = 1, inplace = True)
xtrain.drop('target', axis = 1, inplace = True)

xtest = pd.read_csv(projPath + 'input/xfull_'+ dataset_version + '.csv')
id_test = xtest.ID
xtest.drop('ID', axis = 1, inplace = True)

# Lets develop all interactions of the top N vars.
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

xtrain = poly.fit_transform(xtrain)
xtest = poly.fit_transform(xtest)

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy2(rf, n_estimators=250, verbose=2)

# find all relevant features
feat_selector.fit(xtrain, ytrain)

# check selected features
feat_selector.support_

# check ranking of features
feat_selector.ranking_

