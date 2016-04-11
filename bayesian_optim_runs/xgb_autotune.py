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

    clf.fit(x0, y0, eval_metric="logloss", eval_set=[(x1, y1)],verbose=True,early_stopping_rounds=25)
    ll = -log_loss(y1, clf.predict_proba(x1))
    return ll

if __name__ == "__main__":

    # settings
    projPath = os.getcwd()
    dataset_version = "0411k100"
    todate = datetime.datetime.now().strftime("%Y%m%d")
    input_folder = 'input2'

    ## data
    # read the training and test sets
    xtrain = pd.read_csv('../'+input_folder+'/xtrain_'+ dataset_version + '.csv')
    id_train = xtrain.ID; xtrain.drop('ID', axis = 1, inplace = True)
    ytrain = xtrain.target; xtrain.drop('target', axis = 1, inplace = True)
    xtest = pd.read_csv('../'+input_folder+'/xtest_'+ dataset_version + '.csv')
    id_test = xtest.ID; xtest.drop('ID', axis = 1, inplace = True)

    # folds
    xfolds = pd.read_csv('../input/xfolds2.csv')
    # work with validation split
    idx0 = xfolds[xfolds.valid == 0].index
    idx1 = xfolds[xfolds.valid == 1].index
    x0 = xtrain[xtrain.index.isin(idx0)]
    x1 = xtrain[xtrain.index.isin(idx1)]
    y0 = ytrain[ytrain.index.isin(idx0)]
    y1 = ytrain[ytrain.index.isin(idx1)]

    xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (int(2), int(25)),
                                      'learning_rate': (0.0005, 0.06),
                                      'n_estimators': (int(500), int(2000)),
                                      'subsample': (0.1, 0.99),
                                      'colsample_bytree': (0.1, 0.99),
                                      'gamma': (0.00000000001, 0.05),
                                      'min_child_weight': (int(1), int(40))
                                     })
    xgboostBO.maximize(init_points=5, n_iter=20, acq='ei')
    print('-' * 53)

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])
    print('XGBOOST: %s' % xgboostBO.res['max']['max_params'])
