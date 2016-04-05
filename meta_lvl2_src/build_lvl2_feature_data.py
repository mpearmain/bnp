import pandas as pd
import numpy as np
import datetime
import re
from sklearn.ensemble import RandomForestClassifier
from itertools import permutations, combinations


def build_new_features(xtrain, xtest, top_feats):
    """ Generate 2way new features based on a list of feature names.
    The idea is that we can apply a variety of feature engineering techniques that may give more accurate model
    predictions, in this case we look to take the sum, difference, product, and quotient of the v# variables.
    :param train: A pandas data frame object of training data
    :param test: A pandas data frame object of test data
    :param top_feats: A list of variable names.
    :return: A pandas data frame object with new features developed to the original data frame.
    """
    train = xtrain.copy()
    test = xtest.copy()

    diffquot2way = list(permutations(top_feats, 2))
    sumprod2way = list(combinations(top_feats, 2))
    for A, B in diffquot2way:
        subdiff = "SUB".join([A, B])
        quotient = "DIV".join([A, B])
        train[subdiff] = train[A] - train[B]
        test[subdiff] = test[A] - test[B]
        train[quotient] = train[A] / (1e-15 + train[B])
        test[quotient] = test[A] / (1e-15 + test[B])

    for A, B in sumprod2way:
        addsum = "ADD".join([A, B])
        prods = "PROD".join([A, B])
        train[addsum] = train[A] + train[B]
        test[addsum] = test[A] + test[B]
        train[prods] = train[A] * train[B]
        test[prods] = test[A] * test[B]

    for A in top_feats:
        sqcentre = "SQCentre".join([A])
        train[sqcentre] = (train[A] * train[A]) - train[A]
        train[sqcentre] = (test[A] * test[A]) - test[A]

    return train, test

# We need to import all the meta based predictions from ./metafeatures and
# combine these into a single pandas dataframe.

# settings
projPath = './'
dataset_version = "lvl220160404"
todate = datetime.datetime.now().strftime("%Y%m%d")
# Top fetures to develop meta more interactions variables.
topNfeatures = 10

## data
# read the training and test sets
print "Loading train data."
xtrain = pd.read_csv(projPath + 'input2/xtrain_'+ dataset_version + '.csv')
id_train = xtrain.ID
ytrain = xtrain.target
xtrain.drop('ID', axis = 1, inplace = True)
xtrain.drop('target', axis = 1, inplace = True)
print "Loading test data."
xtest = pd.read_csv(projPath + 'input2/xtest_'+ dataset_version + '.csv')
id_test = xtest.ID
xtest.drop('ID', axis = 1, inplace = True)

# Check the shape of the data is the same
assert xtest.shape[1] == xtrain.shape[1]

# Use Sklearn feature importance to select the 'best' features from our
# metafeatures.

base_feat = list(xtrain)
xgb_feat = filter(lambda x:re.match(r'^xgb',x), base_feat)
noxgb_feat = [x for x in base_feat if x not in xgb_feat]

# First Build a forest and compute the feature importance
forest = RandomForestClassifier(n_jobs=-1,
                                class_weight='auto',
                                max_depth=7,
                                n_estimators=500)
print "Building RF - XGB"
forest.fit(xtrain[xgb_feat], ytrain)
importances = forest.feature_importances_
# Select most important features
indices = np.argsort(importances)[::-1]
top_n_feature_names = list(list(xtrain[xgb_feat])[i] for i in indices[:topNfeatures])
print top_n_feature_names

xtrain, xtest = build_new_features(xtrain, xtest, top_n_feature_names)

print "Shape train", xtrain.shape[1]
print "Shape test", xtest.shape[1]

print "Building RF - Not XGB Features"
forest.fit(xtrain[noxgb_feat], ytrain)
importances = forest.feature_importances_
# Select most important features
indices = np.argsort(importances)[::-1]
top_n_feature_names = list(list(xtrain[noxgb_feat])[i] for i in indices[:topNfeatures])
print top_n_feature_names

xtrain, xtest = build_new_features(xtrain, xtest, top_n_feature_names)

print "Shape train", xtrain.shape[1]
print "Shape test", xtest.shape[1]
# Lets add some simple features.
# This counts the number of meata features that predict target > 0.9
xtrain['gt9'] = (xtrain[base_feat] > 0.9).sum(1)
xtest['gt9'] = (xtest[base_feat] > 0.9).sum(1)
xtrain['gt99'] = (xtrain[base_feat] > 0.99).sum(1)
xtest['gt99'] = (xtest[base_feat] > 0.99).sum(1)

xtrain['lt01'] = (xtrain[base_feat] < 0.1).sum(1)
xtest['lt01'] = (xtest[base_feat] < 0.1).sum(1)
xtrain['lt001'] = (xtrain[base_feat] < 0.01).sum(1)
xtest['lt001'] = (xtest[base_feat] < 0.01).sum(1)

print "Shape train", xtrain.shape[1]
print "Shape test", xtest.shape[1]


xtrain['ID'] = id_train
xtrain['target'] = ytrain
xtest['ID'] = id_test


print 'Writing Data Files.'
xtrain.to_csv("./input2/xtrain_lvl2MP.csv", index = False, header = True)
xtest.to_csv("./input2/xtest_lvl2MP.csv", index = False, header = True)
