import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures

# We need to import all the meta based predictions from ./metafeatures and
# combine these into a single pandas dataframe.

# settings
projPath = './'
dataset_version = "ensemble_base"
todate = datetime.datetime.now().strftime("%Y%m%d")

## data
# read the training and test sets
print "Loading train data."
xtrain = pd.read_csv(projPath + 'input/xvalid_'+ dataset_version + '.csv')
id_train = xtrain.ID
ytrain = xtrain.target
xtrain.drop('ID', axis = 1, inplace = True)
xtrain.drop('target', axis = 1, inplace = True)
print "Loading test data."
xtest = pd.read_csv(projPath + 'input/xfull_'+ dataset_version + '.csv')
id_test = xtest.ID
xtest.drop('ID', axis = 1, inplace = True)

# Use Sklearn feature importance to select the 'best' features from our
# metafeatures.

# First Build a forest and compute the feature importance
forest = RandomForestClassifier(n_jobs=-1,
                                class_weight='auto',
                                max_depth=15,
                                n_estimators=500)
print "Building RF"
forest.fit(xtrain, ytrain)
# Select most important features
model = SelectFromModel(forest, prefit=True)
xtrain_importance = model.transform(xtrain)
xtest_importance = model.transform(xtest)

# Develop all interactions of the top N vars.
print "Building Polynomial interactions"
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
xtrain_importance = pd.DataFrame(poly.fit_transform(xtrain_importance))
xtest_importance = pd.DataFrame(poly.fit_transform(xtest_importance))

print "Combine Base and interactions"
train = pd.concat((xtrain,xtrain_importance), axis=1)
test = pd.concat((xtest,xtest_importance), axis=1)
train['ID'] = id_train
train['target'] = ytrain
test['ID'] = id_test

train.to_csv("./input/xtrain_secondLvL_meta.csv", index = False, header = True)
test.to_csv("./input/xtest_secondLvL_meta.csv", index = False, header = True)

