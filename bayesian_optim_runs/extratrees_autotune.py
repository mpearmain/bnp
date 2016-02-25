from __future__ import division
from __future__ import print_function

__author__ = 'michael.pearmain'

import pandas as pd
import datetime
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from bayesian_optimization import BayesianOptimization
import os

def extratreescv(n_estimators,
                 min_samples_split,
                 min_samples_leaf):

    clf = ExtraTreesClassifier(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               min_samples_leaf=int(min_samples_leaf),
                               max_features="auto",
                               n_jobs=-1,
                               random_state=1234,
                               verbose=0)

    clf.fit(x0, y0)
    ll = -log_loss(y1, clf.predict_proba(x1)[:,1])
    return ll

if __name__ == "__main__":

    # settings
    projPath = os.getcwd()
    dataset_version = "kb1"
    todate = datetime.datetime.now().strftime("%Y%m%d")
    no_bags = 1

    ## data
    # read the training and test sets
    xtrain = pd.read_csv('./input/xtrain_'+ dataset_version + '.csv')
    id_train = xtrain.ID
    ytrain = xtrain.target
    xtrain.drop('ID', axis = 1, inplace = True)
    xtrain.drop('target', axis = 1, inplace = True)
    xtest = pd.read_csv('./input/xtest_'+ dataset_version + '.csv')
    id_test = xtest.ID
    xtest.drop('ID', axis = 1, inplace = True)

    # folds
    xfolds = pd.read_csv('./input/xfolds.csv')
    # work with validation split
    idx0 = xfolds[xfolds.valid == 0].index
    idx1 = xfolds[xfolds.valid == 1].index
    x0 = xtrain[xtrain.index.isin(idx0)]
    x1 = xtrain[xtrain.index.isin(idx1)]
    y0 = ytrain[ytrain.index.isin(idx0)]
    y1 = ytrain[ytrain.index.isin(idx1)]

    extratreesBO = BayesianOptimization(extratreescv,
                                        {'n_estimators': (int(250), int(2000)),
                                         'min_samples_split': (int(2), int(6)),
                                         'min_samples_leaf': (int(1), int(6))
                                         })

    extratreesBO.maximize(init_points=5, n_iter=25, acq='ei')
    print('-' * 53)

    print('Final Results')
    print('Extra Trees: %f' % extratreesBO.res['max']['max_val'])
