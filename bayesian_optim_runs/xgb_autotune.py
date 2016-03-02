from __future__ import division
from __future__ import print_function

__author__ = 'michael.pearmain'

import pandas as pd
import datetime
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from bayes_opt import BayesianOptimization
import os

def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              subsample,
              colsample_bytree,
              gamma,
              min_child_weight,
              silent=True,
              nthread=-1,
              seed=1234):

    clf = XGBClassifier(max_depth=int(max_depth),
                        learning_rate=learning_rate,
                        n_estimators=int(n_estimators),
                        silent=silent,
                        nthread=nthread,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        gamma=gamma,
                        min_child_weight = min_child_weight,
                        seed=seed,
                        objective="binary:logistic")

    clf.fit(x0, y0, eval_metric="logloss", eval_set=[(x1, y1)])
    ll = -log_loss(y1, clf.predict_proba(x1)[:,1])
    return ll

if __name__ == "__main__":

    # settings
    projPath = os.getcwd()
    dataset_version = "kb1"
    todate = datetime.datetime.now().strftime("%Y%m%d")

    ## data
    # read the training and test sets
    xtrain = pd.read_csv('../input/xtrain_'+ dataset_version + '.csv')
    id_train = xtrain.ID
    ytrain = xtrain.target
    xtrain.drop('ID', axis = 1, inplace = True)
    xtrain.drop('target', axis = 1, inplace = True)
    xtest = pd.read_csv('../input/xtest_'+ dataset_version + '.csv')
    id_test = xtest.ID
    xtest.drop('ID', axis = 1, inplace = True)

    # folds
    xfolds = pd.read_csv('../input/xfolds.csv')
    # work with validation split
    idx0 = xfolds[xfolds.valid == 0].index
    idx1 = xfolds[xfolds.valid == 1].index
    x0 = xtrain[xtrain.index.isin(idx0)]
    x1 = xtrain[xtrain.index.isin(idx1)]
    y0 = ytrain[ytrain.index.isin(idx0)]
    y1 = ytrain[ytrain.index.isin(idx1)]

    xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (int(9), int(15)),
                                      'learning_rate': (0.001, 0.02),
                                      'n_estimators': (int(1000), int(2000)),
                                      'subsample': (0.7, 0.9),
                                      'colsample_bytree': (0.7, 0.9),
                                      'gamma': (0.000001, 0.01),
                                      'min_child_weight': (int(5), int(15))
                                     })
    xgboostBO.maximize(init_points=5, n_iter=15, acq='ei')
    print('-' * 53)

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])
    print('XGBOOST: %f' % xgboostBO.res['max']['max_params'])
