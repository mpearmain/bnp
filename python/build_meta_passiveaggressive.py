from __future__ import division
from __future__ import print_function

"""
This metas creation script runs as master template for bayesian optimization and meta level staking for a single
classifier type.

The structure is:
 1. Run a bayesian optimization for the 'best' parameter combinations for a classifier for a dataset.
 2. Set the parameters for a classifier model based off 1.
 3. Run a stacking instance of the classifier
 4. Write a file which contains the meta level information and test meta data.
 5. Repeat for each dataset.

"""

import numpy as np
import pandas as pd
import os
import datetime
from sklearn.linear_model import PassiveAggressiveClassifier
from python.BinaryStacker import BinaryStackingClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import log_loss


def passive_aggressive(C, n_iter, loss_metric=log_loss, maximize=False):

    clf = PassiveAggressiveClassifier(C=C,
                                      fit_intercept=True,
                                      n_iter=n_iter,
                                      shuffle=True,
                                      verbose=0,
                                      loss="hinge",
                                      n_jobs=-1,
                                      random_state=random_seed)

    clf.fit(x0, y0)
    if maximize:
        loss = loss_metric(y1, clf.predict_proba(x1)[:,1])
    if not maximize:
        loss = -loss_metric(y1, clf.predict_proba(x1)[:,1])
    return loss


if __name__ == '__main__':
    ## settings
    projPath = os.getcwd()
    dataset_version = ["kb1", "kb2", "kb3", "kb4", "kb5099", "kb6099"]
    model_type = "PassiveAggressive"
    todate = datetime.datetime.now().strftime("%Y%m%d")
    random_seed = 1234

    # construct colnames for the data  - Only alpha numerics as xgboost in
    # second level metas doesnt like special chars.
    clfnames = [model_type + str(random_seed) + str(dataset_version[n]) for n in range(len(dataset_version))]
    # Setup pandas dataframe to store full result in.
    train = pd.read_csv(projPath + '/input/train.csv')
    test = pd.read_csv(projPath + '/input/test.csv')

    # Create Data Frames to store results.
    mvalid = pd.DataFrame(np.nan, index=train.index, columns=clfnames)
    mfull = pd.DataFrame(np.nan, index=test.index, columns=clfnames)

    del train, test

    ########################## Run Bayesian optimization per dataset ####################################

    for i, dataset in enumerate(dataset_version):
        print('\nRunning Bayes for Dataset', dataset)
        # read the training and test sets
        xtrain = pd.read_csv(projPath + '/input/xtrain_' + dataset + '.csv')
        id_train = xtrain.ID
        ytrain = xtrain.target
        xtrain.drop('ID', axis = 1, inplace = True)
        xtrain.drop('target', axis = 1, inplace = True)
        xtest = pd.read_csv(projPath + '/input/xtest_'+ dataset + '.csv')
        id_test = xtest.ID
        xtest.drop('ID', axis = 1, inplace = True)

        # folds
        xfolds = pd.read_csv(projPath + '/input/xfolds.csv')
        # work with validation split
        idx0 = xfolds[xfolds.valid == 0].index
        idx1 = xfolds[xfolds.valid == 1].index
        x0 = xtrain[xtrain.index.isin(idx0)]
        x1 = xtrain[xtrain.index.isin(idx1)]
        y0 = ytrain[ytrain.index.isin(idx0)]
        y1 = ytrain[ytrain.index.isin(idx1)]

        BO = BayesianOptimization(passive_aggressive,
                                  {'C': (0.2, 30.),
                                   'n_iter':(int(5), int(50))
                                   })

        BO.maximize(init_points=5, n_iter=15, acq='ei')
        print('-' * 53)

        print('Final Results')
        print('Extra Trees Loss: %f' % BO.res['max']['max_val'])
        print('Extra Trees Params: %s' % BO.res['max']['max_params'])

        del idx0, idx1, x0, x1, y0, y1

        ########################## Run Best model per dataset ####################################

        clf = [PassiveAggressiveClassifier(BO.res['max']['max_params']['C'],
                                           fit_intercept=True,
                                           n_iter=BO.res['max']['max_params']['n_iter'],
                                           shuffle=True,
                                           verbose=0,
                                           loss="hinge",
                                           n_jobs=-1,
                                           random_state=random_seed)]

        # Read xfolds only need the ID and fold 5.
        print("Reading Cross folds")
        xfolds = pd.read_csv(projPath + '/input/xfolds.csv', usecols=['ID','fold5'])

        print("Running Stacker")
        stacker = BinaryStackingClassifier(base_classifiers=clf,
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