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
    random_seed = 42
    todate = datetime.datetime.now().strftime("%Y%m%d")

    # construct colnames for the data  - Only alpha numerics as xgboost in
    # second level metas doesnt like special chars.
    clfnames = [model_type + str(random_seed) + str(dataset_version[n])
                for n in range(len(dataset_version))]

    # setup model instances
    clf = [ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators= , min_samples_split= , min_samples_leaf=),
           ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1665, min_samples_split=2 , min_samples_leaf=2),
           ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1784, min_samples_split=5 , min_samples_leaf=3),
           ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1258, min_samples_split=4 , min_samples_leaf=2),
           ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1972, min_samples_split=3 , min_samples_leaf=2),
           ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1577 , min_samples_split=4 , min_samples_leaf=2),
           ExtraTreesClassifier(n_jobs= -1, random_state= random_seed, n_estimators=1370 , min_samples_split=2 , min_samples_leaf=3)]


    # Read xfolds only need the ID and fold 5.
    print("Reading Cross folds")
    xfolds = pd.read_csv(projPath + '/input/xfolds.csv', usecols=['ID','fold5'])

    # read the training and test sets
    print("Reading Train set")
    train = pd.read_csv(projPath + '/input/train.csv')
    print("Reading Test set")
    test = pd.read_csv(projPath + '/input/test.csv')

    # Setup pandas dataframe to store full result in.
    mvalid = pd.DataFrame(np.nan, index=train.index, columns=clfnames)
    mfull = pd.DataFrame(np.nan, index=test.index, columns=clfnames)

    # Create the loops over the datasets.
    for i, dataset in enumerate(dataset_version):
        # read the training and test sets
        print("Reading Train set", dataset)
        xtrain = pd.read_csv(projPath + '/input/xtrain_'+ dataset + '.csv')
        id_train = xtrain.ID
        ytrain = xtrain.target
        xtrain.drop('ID', axis = 1, inplace = True)
        xtrain.drop('target', axis = 1, inplace = True)
        print("Reading Test set")
        xtest = pd.read_csv(projPath + '/input/xtest_'+ dataset + '.csv')
        id_test = xtest.ID
        xtest.drop('ID', axis = 1, inplace = True)

        stacker = BinaryStackingClassifier(base_classifiers=clf[i],
                                           xfolds=xfolds,
                                           evaluation=log_loss,
                                           colnames=clfnames[i])
        stacker.fit(xtrain, ytrain)

        # Append the results for each dataset back to the master for train and test











        # store the results
        # add indices etc
        mvalid = pd.DataFrame(mvalid)
        mvalid.columns = [model_type + str(i) for i in range(0, mvalid.shape[1])]
        mvalid['ID'] = id_train
        mvalid['target'] = ytrain

        mfull = pd.DataFrame(mfull)
        mfull.columns = [model_type + str(i) for i in range(0, mfull.shape[1])]
        mfull['ID'] = id_test


        # save the files
        mvalid.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
        mfull.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)

        # Save params
        params = pd.DataFrame(param_grid)
        params.to_csv(projPath + 'meta_parameters/' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False)