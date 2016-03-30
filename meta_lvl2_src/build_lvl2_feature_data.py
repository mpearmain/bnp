import pandas as pd
import numpy as np
import datetime
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

    return train, test

# We need to import all the meta based predictions from ./metafeatures and
# combine these into a single pandas dataframe.

# settings
projPath = '../'
dataset_version = "lvl220160330"
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

# Use Sklearn feature importance to select the 'best' features from our
# metafeatures.

# First Build a forest and compute the feature importance
forest = RandomForestClassifier(n_jobs=-1,
                                class_weight='auto',
                                max_depth=20,
                                n_estimators=1000)
print "Building RF"
forest.fit(xtrain, ytrain)
importances = forest.feature_importances_
# Select most important features
indices = np.argsort(importances)[::-1]
top_n_feature_names = list(list(xtrain)[i] for i in indices[:topNfeatures])
print top_n_feature_names

xtrain, xtest = build_new_features(xtrain, xtest, top_n_feature_names)

xtrain['ID'] = id_train
xtrain['target'] = ytrain
xtest['ID'] = id_test

print 'Writing Data Files.'
xtrain.to_csv("./input/xtrain_secondLvL_meta.csv", index = False, header = True)
xtest.to_csv("./input/xtest_secondLvL_meta.csv", index = False, header = True)
