# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:08:00 2015

@author: konrad
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from itertools import product
import datetime

if __name__ == '__main__':

    ## settings
    dataset_version = "lvl220160331combo"
    model_type = "etrees" 
    seed_value = 789
    todate = datetime.datetime.now().strftime("%Y%m%d")
    	    
    ## data
    xtrain = pd.read_csv('../input2/xtrain_'+ dataset_version + '.csv')     
    id_train = xtrain.ID
    y = xtrain.target
    xtrain.drop('target', axis = 1, inplace = True)
    xtrain.drop('ID', axis = 1, inplace = True)

    xtest = pd.read_csv('../input2/xtest_'+ dataset_version + '.csv')     
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
    model = ExtraTreesClassifier(n_estimators=10, 
                                 criterion='gini', 
                                 max_depth=None, 
                                 min_samples_split=2, 
                                 min_samples_leaf=1, 
                                 min_weight_fraction_leaf=0.0, 
                                 max_features='auto', 
                                 max_leaf_nodes=None, 
                                 n_jobs=4, 
                                 random_state=seed_value, 
                                 class_weight=None)

    # parameter grids    
    ntree_vals = [250]
    maxdepth_vals = [15,35]
    minsampsplit_vals = [2, 10]
    minsampleaf_vals = [1, 10]
    mwfl_vals = [ 0.005]
    maxfeat_vals = [25, 50]
    classweight = ['balanced_subsample']
    param_grid = tuple([ntree_vals, maxdepth_vals, minsampsplit_vals, 
                        minsampleaf_vals, mwfl_vals, maxfeat_vals, classweight])
    param_grid = list(product(*param_grid))

    # (dataset version, model type, seed, parameter grid) 
#    par_dump = '../meta_parameters/'+'D'+dataset_version+'_M'+model_type  
#    par_dump = par_dump + '_'+todate+'.txt'
#    f1=open(par_dump, 'w+')
#    f1.write('dataset version: '); f1.write(str(dataset_version))
#    f1.write('\nmodel type:'); f1.write(str(model_type))
#    f1.write('\nseed value: '); f1.write(str(seed_value))    
#    f1.write('\nparameter grid \n'); f1.write(str(param_grid)    )
#    f1.close()
    
    # storage structure for forecasts
    mvalid = np.zeros((xtrain.shape[0],len(param_grid)))
    mfull = np.zeros((xtest.shape[0],len(param_grid)))
    
    ## build 2nd level forecasts
    for i in range(len(param_grid)):        
            print "processing parameter combo:", i
            # configure model with j-th combo of parameters
            x = param_grid[i]
            model.n_estimators = x[0]
            model.max_depth = x[1]     
            model.min_samples_split = x[2]
            model.min_samples_leaf = x[3]
            model.min_weight_fraction_leaf = x[4]
            model.max_features = x[5]
            model.class_weight = x[6]
            
            # loop over folds
            for j in range(0,n_folds):
                idx0 = np.where(fold_index != j)
                idx1 = np.where(fold_index == j)
                x0 = np.array(xtrain)[idx0,:][0];
                x1 = np.array(xtrain)[idx1,:][0]
                y0 = np.array(y)[idx0];
                y1 = np.array(y)[idx1]

                model.fit(x0, y0)
                y_pre = model.predict_proba(x1)[:,1]
                mvalid[idx1,i] = y_pre
                print 'log loss: ', log_loss(y1,y_pre)
                print "finished fold:", j
                
            # fit on complete dataset
            model.fit(xtrain, y)
            mfull[:,i] = model.predict_proba(xtest)[:,1]
            print "finished full prediction"
            
    ## store the results
    # add indices etc
    mvalid = pd.DataFrame(mvalid)
    mvalid.columns = [model_type + str(i) for i in range(0, mvalid.shape[1])]
    mvalid['ID'] = id_train
    mvalid['target'] = y
    
    mfull = pd.DataFrame(mfull)
    mfull.columns = [model_type + str(i) for i in range(0, mfull.shape[1])]
    mfull['ID'] = id_test
    

    # save the files            
    mvalid.to_csv('../metafeatures2/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    mfull.to_csv('../metafeatures2/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    
