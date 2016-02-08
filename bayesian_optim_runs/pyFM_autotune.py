from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import datetime
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from fastFM import sgd
from sklearn.metrics import log_loss
from bayesian_optimization import BayesianOptimization


def FMcv(n_iter,
         init_stdev,
         l2_reg_w,
         l2_reg_V,
         rank,
         step_size):

    fm = sgd.FMClassification(n_iter=n_iter,
                              init_stdev=init_stdev,
                              l2_reg_w=l2_reg_w,
                              l2_reg_V=l2_reg_V,
                              rank=rank,
                              step_size=step_size)
    fm.fit(x0, y0)
    ll = -log_loss(y1, fm.predict_proba(x0)[:,1])
    return ll

if __name__ == "__main__":

    # settings
    projPath = './'
    dataset_version = "mp1"
    todate = datetime.datetime.now().strftime("%Y%m%d")
    no_bags = 1

    ## data
    # read the training and test sets
    xtrain = pd.read_csv(projPath + 'input/xtrain_'+ dataset_version + '.csv')
    id_train = xtrain.ID
    ytrain = xtrain.target
    xtrain.drop('ID', axis = 1, inplace = True)
    xtrain.drop('target', axis = 1, inplace = True)

    xtest = pd.read_csv(projPath + 'input/xtest_'+ dataset_version + '.csv')
    id_test = xtest.ID
    xtest.drop('ID', axis = 1, inplace = True)

    # folds
    xfolds = pd.read_csv(projPath + 'input/xfolds.csv')
    # work with validation split
    idx0 = xfolds[xfolds.valid == 0].index
    idx1 = xfolds[xfolds.valid == 1].index
    x0 = csc_matrix((xtrain[xtrain.index.isin(idx0)]).as_matrix())
    x1 = csc_matrix((xtrain[xtrain.index.isin(idx1)]).as_matrix())
    y0 = ytrain[ytrain.index.isin(idx0)].as_matrix()
    y0[y0 == 0] = -1
    y1 = ytrain[ytrain.index.isin(idx1)].as_matrix()
    y1[y1 == 0] = -1
    FMcvBO = BayesianOptimization(FMcv,
                                     {'n_iter': (int(500), int(1000)),
                                      'init_stdev': (0.05, 0.2),
                                      'l2_reg_w': (0.0, 0.000001),
                                      'l2_reg_V': (0.0, 0.000001),
                                      'rank': (int(8), int(20)),
                                      'step_size': (0.2, 0.05)
                                     })

    FMcvBO.maximize(init_points=7, restarts=500, n_iter=25)
    print('-' * 53)

    print('Final Results')
    print('Factorization Machines: %f' % FMcvBO.res['max']['max_val'])
