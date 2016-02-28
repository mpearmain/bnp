from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import datetime
import os
from python.BinaryStacker import BinaryStackingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss

if __name__ == '__main__':

    ## settings
    projPath = os.getcwd()
    dataset_version = ["mp1", "kb1", "kb2", "kb3", "kb4", "kb5099", "kb6099"]
    model_type = "etrees"
    # Generate the same random sequences.
    random_seed = 1234
    todate = datetime.datetime.now().strftime("%Y%m%d")

    # construct colnames for the data  - Only alpha numerics as xgboost in
    # second level metas doesnt like special chars.
    clfnames = [model_type + str(random_seed) + str(dataset_version[n])
                for n in range(len(dataset_version))]

    # setup model instances
    clf = [ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1812, min_samples_split=3 , min_samples_leaf=1),
           ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1665, min_samples_split=2 , min_samples_leaf=2),
           ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1784, min_samples_split=5 , min_samples_leaf=3),
           ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1258, min_samples_split=4 , min_samples_leaf=2),
           ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1972, min_samples_split=3 , min_samples_leaf=2),
           ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1577, min_samples_split=4 , min_samples_leaf=2),
           ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1370, min_samples_split=2 , min_samples_leaf=3)]


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
        stacker.fit(xtrain, ytrain)

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