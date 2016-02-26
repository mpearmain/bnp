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
    dataset_version = ["kb1", "kb3", "kb4", "kb5099", "kb6099"]
    model_type = "etrees"
    # Generate the same random sequences.
    random_seed = 1234
    todate = datetime.datetime.now().strftime("%Y%m%d")

    # construct colnames for the data  - Only alpha numerics as xgboost in
    # second level metas doesnt like special chars.
    clfnames = [model_type + str(random_seed) + str(dataset_version[n])
                for n in range(len(dataset_version))]

    # setup model instances
    clf = [XGBClassifier(max_depth=11, learning_rate=0.01, n_estimators=1962, silent=True, nthread=-1, subsample=0.80883233339510385, colsample_bytree=0.90000000000000002, gamma=0.0001, min_child_weight = 1, seed=random_seed, objective="binary:logistic"),
           XGBClassifier(max_depth=11, learning_rate=0.0092528248736576668, n_estimators=1906, silent=True, nthread=-1, subsample=0.89010324821493381, colsample_bytree=0.89095719586675526, gamma=0.0045373086289034713, min_child_weight = 5, seed=random_seed, objective="binary:logistic"),
           XGBClassifier(max_depth=14, learning_rate=0.0073362638967263945, n_estimators=2408, silent=True, nthread=-1, subsample= 0.72679682406267243, colsample_bytree=0.76427399221822834, gamma=0.0071936123399884092, min_child_weight = 14, seed=random_seed, objective="binary:logistic"),
           XGBClassifier(max_depth=10, learning_rate=0.025, n_estimators=2500, silent=True, nthread=-1, subsample=0.9, colsample_bytree=0.69999999999999996, gamma=0.00077979306474894653, min_child_weight = 1, seed=random_seed, objective="binary:logistic"),
           # Waiting for best kb6099 params
           XGBClassifier(max_depth=11, learning_rate=0.01, n_estimators=1962, silent=True, nthread=-1, subsample=0.80883233339510385, colsample_bytree=0.90000000000000002, gamma=0.0001, min_child_weight = 1, seed=random_seed, objective="binary:logistic")
           ]

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
        mvalid.ix[:, i] = stacker.meta_train.ix[:, i]
        mfull.ix[:, i] = stacker.predict_proba(xtest)

    # store the results
    mvalid['ID'] = id_train
    mvalid['target'] = ytrain
    mfull['ID'] = id_test

    # save the files
    mvalid.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_seed' + str(random_seed) + '.csv', index = False, header = True)
    mfull.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_seed' + str(random_seed) + '.csv', index = False, header = True)