# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 00:20:32 2015

@author: konrad
"""

import numpy as np
import pandas as pd
import datetime
import xgboost as xgb
import os


if __name__ == '__main__':

    ## settings
    projPath = os.getcwd()
    dataset_version = "kb1"
    model_type = "xgb"
    seed_value = 889
    todate = datetime.datetime.now().strftime("%Y%m%d")

    ## data
    # read the training and test sets
    xtrain = pd.read_csv('../input/xtrain_'+ dataset_version + '.csv')
    id_train = xtrain.QuoteNumber
    ytrain = xtrain.QuoteConversion_Flag
    xtrain.drop('QuoteNumber', axis = 1, inplace = True)
    xtrain.drop('QuoteConversion_Flag', axis = 1, inplace = True)

    xtest = pd.read_csv('../input/xtest_'+ dataset_version + '.csv')
    id_test = xtest.QuoteNumber
    xtest.drop('QuoteNumber', axis = 1, inplace = True)

    # folds
    xfolds = pd.read_csv('../input/xfolds.csv')
    # work with 5-fold split
    fold_index = xfolds.fold5
    fold_index = np.array(fold_index) - 1
    n_folds = len(np.unique(fold_index))

    ## model
    '''
    index convention same as BO:
        colsample_bytree : 0
        learning_rate    : 1
        min_child_weight : 2
        n_estimators     : 3
        subsample        : 4
        max_depth        : 5
        gamma            : 6
    '''
    
    param_grid = [
         (0.7,0.0090705965482161376,5.0,1341,0.7,13, 0.01),
         (0.7,0.02,5.0,650,0.7, 7, 0.01),
         (0.8,0.0090705965482161376,5.0,1341,0.8,13, 0.001)                      
    ]

    # dump the meta description for this set into a file
    # (dataset version, model type, seed, parameter grid) 
    par_dump = '../meta_parameters/'+'D'+dataset_version+'_M'+model_type  
    par_dump = par_dump + '_'+todate+'.txt'
    f1=open(par_dump, 'w+')
    f1.write('dataset version: '); f1.write(str(dataset_version))
    f1.write('\nmodel type:'); f1.write(str(model_type))
    f1.write('\nseed value: '); f1.write(str(seed_value))    
    f1.write('\nparameter grid \n'); f1.write(str(param_grid)    )
    f1.close()
    
    # storage structure for forecasts
    mvalid = np.zeros((xtrain.shape[0],len(param_grid)))
    mfull = np.zeros((xtest.shape[0],len(param_grid)))

    ## build 2nd level forecasts
    for i in range(len(param_grid)):
        print "processing parameter combo:", param_grid[i]
        # configure model with j-th combo of parameters
        x = param_grid[i]
        clf = xgb.XGBClassifier(nthread=-1,
                                seed=seed_value,
                                silent=True,                                
                                colsample_bytree=x[0],
                                learning_rate=x[1],
                                min_child_weight=x[2],
                                n_estimators=x[3],                                
                                subsample=x[4],
                                max_depth=x[5],                                
                                gamma=x[6]
                                )

        # loop over folds - Keeping as pandas for ease of use with xgb wrapper
        for j in range(1 ,n_folds+1):
            idx0 = xfolds[xfolds.fold5 != j].index
            idx1 = xfolds[xfolds.fold5 == j].index
            x0 = xtrain[xtrain.index.isin(idx0)]
            x1 = xtrain[xtrain.index.isin(idx1)]
            y0 = ytrain[ytrain.index.isin(idx0)]
            y1 = ytrain[ytrain.index.isin(idx1)]

            # fit the model on observations associated with subject whichSubject in this fold
            clf.fit(x0, y0, eval_metric="logloss", eval_set=[(x1, y1)])
            mvalid[idx1,i] = clf.predict_proba(x1)[:,1]

        # fit on complete dataset
        bst = xgb.XGBClassifier(
                                nthread=-1,
                                seed=seed_value,
                                silent=True,                                
                                colsample_bytree=x[0],
                                learning_rate=x[1],
                                min_child_weight=x[2],
                                n_estimators=x[3],                                
                                subsample=x[4],
                                max_depth=x[5],                                
                                gamma=x[6])
        bst.fit(xtrain, ytrain, eval_metric="logloss")
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
    mvalid.to_csv('../metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    mfull.to_csv('../metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)