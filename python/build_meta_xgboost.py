# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
import xgboost as xgb
from sklearn.metrics import log_loss

if __name__ == '__main__':

    ## settings
    projPath = './'
    dataset_version = "mp1"
    model_type = "xgb"
    seed_value = 123
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
    # work with 5-fold split
    fold_index = xfolds.fold5
    fold_index = np.array(fold_index) - 1
    n_folds = len(np.unique(fold_index))

    ## model
    # parameter grids: LR + range of training subjects to subset to
    '''
    Staying with index convention.
        child_weight = 0
        max_depth = 1
        colsample = 2
        rowsample = 3
        gamma_val = 4
        eta_val = 5
        ntrees = 6
    '''
    param_grid = [(1, 7, 0.75, 0.68, 0.0, 0.01, 1800),
                  (1, 20, 0.85, 0.8, 0.00008, 0.015, 232),
                  (4, 5, 0.85, 0.81, 0.000363, 0.017483, 1535),
                  (3, 12, 0.83, 0.885, 0.000247, 0.01970, 446),
                  (1, 10, 0.81, 0.839, 0.000746, 0.01222, 771),
                  (1, 12, 0.845, 0.892, 0.0004782, 0.01877, 216),
                  (1, 6, 0.84, 0.892, 0.000502716, 0.018, 600)
                  ]

    # storage structure for forecasts
    mvalid = np.zeros((xtrain.shape[0],len(param_grid)))
    mfull = np.zeros((xtest.shape[0],len(param_grid)))

    ## build 2nd level forecasts
    for i in range(len(param_grid)):
        print "processing parameter combo:", param_grid[i]
        # configure model with j-th combo of parameters
        x = param_grid[i]
        clf = xgb.XGBClassifier(n_estimators=x[6],
                                nthread=-1,
                                max_depth=x[1],
                                min_child_weight=x[0],
                                learning_rate=x[5],
                                silent=True,
                                subsample=x[3],
                                colsample_bytree=x[2],
                                gamma=x[2],
                                seed=seed_value)

        # loop over folds - Keeping as pandas for ease of use with xgb wrapper
        for j in range(1 ,n_folds+1):
            idx0 = xfolds[xfolds.fold5 != j].index
            idx1 = xfolds[xfolds.fold5 == j].index
            x0 = xtrain[xtrain.index.isin(idx0)]
            x1 = xtrain[xtrain.index.isin(idx1)]
            y0 = ytrain[ytrain.index.isin(idx0)]
            y1 = ytrain[ytrain.index.isin(idx1)]

            # fit the model on observations associated with subject whichSubject in this fold
            clf.fit(x0, y0, eval_metric='logloss', eval_set=[(x1, y1)])
            print 'Logloss on fold:', log_loss(y1, clf.predict_proba(x1)[:,1])
            mvalid[idx1,i] = clf.predict_proba(x1)[:,1]

        # fit on complete dataset
        bst = xgb.XGBClassifier(n_estimators=x[6],
                                nthread=-1,
                                max_depth=x[1],
                                min_child_weight=x[0],
                                learning_rate=x[5],
                                silent=True,
                                subsample=x[3],
                                colsample_bytree=x[2],
                                gamma=x[2],
                                seed=seed_value)
        bst.fit(xtrain, ytrain, eval_metric='logloss')
        mfull[:,i] = bst.predict_proba(xtest)[:,1]


    ## store the results
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
