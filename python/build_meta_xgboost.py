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
    dataset_version = "lvl220160331combo"
    model_type = "xgb"
    seed_value = 308
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
          (0.86939934445906852,0.02992847302109345, 20.631692294719159,
           318, 0.72547208975310051, 10, 0.01966513396708065 ),
          #   (0.59461387900382301, 0.01058200585695002,
          #    8.5193167994862655, 758, 0.7899466628174614, 
          #    14, 0.015413770403341487),
              # optimized for lvl220160329
          #  (0.59449161316455967, 0.03723339549515764,3.8858379480282639, 
          #113,  0.63669942372294597,  9, 0.0064676261299398399),
          #(0.85, 0.02, 1, 500, 0.9, 6, 0),
          #(0.71641653481931888,  0.012967411293674902, 22.080483788936164, 
          # 507, 0.77459872346718606, 8, 0.0083553529050680725),
           # optimized for c20160330
          #(0.95, 0.010802548227853035, 25.0, 438, 0.95, 
          #  6, 9.9999999999999995e-07),
            ## optimized for lvl220160331xgb
            (0.5889440722623932,  0.026336192899721379, 
            23.06860893667988,  186,  0.94485239119170727, 
             8, 0.016613348739321243),
            ## optimized for lvl220160331combo
            (0.94999999999999996,  0.031469810335467224, 
                 30.0, 336,  0.84172191810733488,  6,  0.02)
            # optimized for kb6099num
           (0.79637131133240435, 0.02325932078657638, 
            5.5688598224060568, 512,  0.55562854771368753, 15, 
                0.0041103453838154513)      
             
    ]
    
    # dump the meta description for this set into a file
    # (dataset version, model type, seed, parameter grid) 
    par_dump = '../meta_parameters2/'+'D'+dataset_version+'_M'+model_type  
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
