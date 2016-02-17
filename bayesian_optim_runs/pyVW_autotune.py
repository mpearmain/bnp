from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import datetime
import numpy as np
import pandas as pd
from python.sklearn_vw import VWClassifier


if __name__ == "__main__":

    # settings
    projPath = './'
    dataset_version = "kb1"
    todate = datetime.datetime.now().strftime("%Y%m%d")

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
    x0 = xtrain[xtrain.index.isin(idx0)]
    x1 = xtrain[xtrain.index.isin(idx1)]
    y0 = ytrain[ytrain.index.isin(idx0)]
    # VW expects -1 not 0 values.
    y0[y0==0] = -1
    y1 = ytrain[ytrain.index.isin(idx1)]
    y1[y1==0] = -1

    # Need to be numpy array
    x0 = x0.astype(np.float32).as_matrix()
    y0 = y0.astype(np.float32).as_matrix()
    x1 = x1.astype(np.float32).as_matrix()
    y1 = y1.astype(np.float32).as_matrix()

    # build vowpal wabbit model
    model = VWClassifier(probabilities=None,
                         random_seed=1234,
                         learning_rate=0.15,
                         l=None,
                         power_t=None,
                         decay_learning_rate=None,
                         input_feature_regularizer=None,
                         progress=True,
                         P=None,
                         quiet=False,
                         b=22,
                         min_prediction=1e-15,
                         max_prediction=1-1e-15,
                         loss_function='logistic',
                         quantile_tau=None,
                         l1=2,
                         l2=2,
                         passes=5)
    model.fit(x0, y0)
    model.score(x1,y1)
