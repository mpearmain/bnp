# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:44:27 2015

@author: konrad
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from itertools import product
import datetime
import os
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier


if __name__ == '__main__':

    ## settings
    projPath = os.getcwd()
    dataset_version = "lvl220160331xgb"
    seed_value = 123
    todate = datetime.datetime.now().strftime("%Y%m%d")
    nbag = 100
    model_type = 'bagxgb' + str(nbag)
    
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
    # setup model instances
    model = xgb.XGBClassifier(
                                nthread=-1,
                                seed=seed_value,
                                silent=True)
                                
    param_grid = [
            ## optimized for lvl220160331xgb
            (0.5889440722623932,  0.026336192899721379, 
            23.06860893667988,  186,  0.94485239119170727, 
             8, 0.016613348739321243),
              
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
        
            print "processing parameter combo:", i
            # configure model with j-th combo of parameters
            x = param_grid[i]
            model.colsample_bytree=x[0]; model.learning_rate=x[1]; 
            model.min_child_weight=x[2]; model.n_estimators=x[3]    
            model.subsample=x[4]; model.max_depth=x[5]                               
            model.gamma=x[6]
                                           
            # loop over folds
            for j in range(0,n_folds):
                idx0 = np.where(fold_index != j)
                idx1 = np.where(fold_index == j)
                x0 = np.array(xtrain)[idx0,:][0]; x1 = np.array(xtrain)[idx1,:][0]
                y0 = np.array(ytrain)[idx0]; y1 = np.array(ytrain)[idx1]
		
                # setup bagging classifier
                bag0 = BaggingClassifier(base_estimator=model, 
                        n_estimators=10, 
                        max_samples=0.05, 
                        max_features=0.9, 
                        bootstrap=True, 
                        bootstrap_features=False, 
                        oob_score=False, 
                        warm_start=False, 
                        n_jobs=-1, random_state=seed_value, 
                        verbose=2)
                        
                bag0.fit(x0, y0)
                prx = bag0.predict_proba(x1)[:,1]
                mvalid[idx1,i] = prx
                print log_loss(y1, prx)
                print "finished fold:", j
                
            # fit on complete dataset
            bag0.fit(xtrain, ytrain)
            mfull[:,i] = bag0.predict_proba(xtest)[:,1]
            print "finished full prediction"
        
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
    
