# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import datetime
import os
from python.BinaryStacker import BinaryStackingClassifier
from sklearn.metrics import log_loss

if __name__ == '__main__':

    ## settings
    projPath = os.getcwd()
    dataset_version = ["mp1", "kb2", "kb3", "kb4", "kb5099", "kb6099"]
    model_type = "XGB"
    # Generate the same random sequences.
    random_seed = 1234
    todate = datetime.datetime.now().strftime("%Y%m%d")

    # construct colnames for the data  - Only alpha numerics as xgboost in
    # second level metas doesnt like special chars.
    clfnames = [model_type + str(random_seed) + str(dataset_version[n])
                for n in range(len(dataset_version))]

    # setup model instances
    clf = [XGBClassifier(max_depth=9, learning_rate=0.0062, n_estimators=1819, silent=True, nthread=-1, subsample=0.7, colsample_bytree=0.7, gamma=0.01, min_child_weight = 5, seed=random_seed, objective="binary:logistic"),
           XGBClassifier(max_depth=12, learning_rate=0.00708, n_estimators=1781, silent=True, nthread=-1, subsample=0.7104, colsample_bytree=0.7, gamma=0.01, min_child_weight = 5.5917305068934944, seed=random_seed, objective="binary:logistic"),
           XGBClassifier(max_depth=11, learning_rate=0.00925, n_estimators=1906, silent=True, nthread=-1, subsample=0.89, colsample_bytree=0.8909, gamma=0.004537, min_child_weight = 5.6470775432075406, seed=random_seed, objective="binary:logistic"),
           XGBClassifier(max_depth=14, learning_rate=0.007336, n_estimators=2408, silent=True, nthread=-1, subsample=0.7267, colsample_bytree=0.7642, gamma=0.00719, min_child_weight = 14.634866816577702, seed=random_seed, objective="binary:logistic"),
           XGBClassifier(max_depth=10, learning_rate=0.025, n_estimators=2500, silent=True, nthread=-1, subsample=0.9, colsample_bytree=0.7, gamma=0.00077979306474894653, min_child_weight = 1.633309904669616, seed=random_seed, objective="binary:logistic"),
           XGBClassifier(max_depth=10, learning_rate=0.013901402098648891, n_estimators=1519, silent=True, nthread=-1, subsample=0.804, colsample_bytree=0.772, gamma=0.00060856204654098059, min_child_weight = 12.326654624743421, seed=random_seed, objective="binary:logistic")]

    # Read xfolds only need the ID and fold 5.
    print("Reading Cross folds")
    xfolds = pd.read_csv(projPath + '/input/xfolds.csv', usecols=['ID','fold5'])

    # read the training and test sets
    print("Reading Train set")
    train = pd.read_csv(projPath + '/input/train.csv')
    id_train = train.ID
    ytrain = train.target
    print("Reading Test set")
    test = pd.read_csv(projPath + '/input/test.csv')
    id_test = test.ID

    # Setup pandas dataframe to store full result in.
    mvalid = pd.DataFrame(np.nan, index=train.index, columns=clfnames)
    mfull = pd.DataFrame(np.nan, index=test.index, columns=clfnames)

    del train, test

    # Create the loops over the datasets.
    for i, dataset in enumerate(dataset_version):
        # read the training and test sets
        print("Reading Train set", dataset)
        xtrain = pd.read_csv(projPath + '/input/xtrain_'+ dataset + '.csv')
        ytrain = xtrain.target
        xtrain.drop('ID', axis = 1, inplace = True)
        xtrain.drop('target', axis = 1, inplace = True)

        print("Reading Test set")
        xtest = pd.read_csv(projPath + '/input/xtest_'+ dataset + '.csv')
        xtest.drop('ID', axis = 1, inplace = True)

        stacker = BinaryStackingClassifier(base_classifiers=[clf[i]],
                                           xfolds=xfolds,
                                           evaluation=log_loss)
        stacker.fit(xtrain, ytrain, eval_metric="logloss")

        # Append the results for each dataset back to the master for train and test
        mvalid.ix[:, i] = stacker.meta_train.ix[:, 0]
        mfull.ix[:, i] = stacker.predict_proba(xtest)

    # store the results
    mvalid['ID'] = id_train
    mvalid['target'] = ytrain
    mfull['ID'] = id_test

    # save the files
    mvalid.to_csv(projPath + '/metafeatures/prval_' + model_type + '_' + todate + '_seed' + str(random_seed) + '.csv', index = False, header = True)
    mfull.to_csv(projPath + '/metafeatures/prfull_' + model_type + '_' + todate + '_seed' + str(random_seed) + '.csv', index = False, header = True)