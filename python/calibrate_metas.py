# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 21:55:16 2016

@author: konrad
"""

from __future__ import division
from __future__ import print_function

__author__ = 'konrad.banachewicz'

import pandas as pd
from sklearn.metrics import log_loss
import os
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression as IR

# from get_diagram_data import get_diagram_data

if __name__ == "__main__":

    # settings
    #projPath = os.getcwd()
    #print(projPath)
    
    # Read xfolds only need the ID and fold 5.
    print("Reading cross folds")
    xfolds = pd.read_csv('../input/xfolds.csv', usecols=['ID','fold5'])
    nfolds = len(np.unique(xfolds.fold5))
    
    # read the metas to calibrate
    print("Reading train set")
    xtrain = pd.read_csv('../metafeatures/prval_XGB_20160227_seed1234.csv')
    id_train = xtrain.ID
    ytrain = xtrain.target
    yval = xtrain.target * 0 -1

    print("Reading test set")
    xtest = pd.read_csv('../metafeatures/prfull_XGB_20160227_seed1234.csv')
    id_test = xtest.ID
    yfull = xtest.ID * 0 -1

    
    storage_mat = np.zeros((nfolds, 3))
    
    # populate the new prval
    for j in range(0,nfolds):
        print("fold: " + str(j))
        idx0 = xfolds[xfolds.fold5 != j + 1].index
        idx1 = xfolds[xfolds.fold5 == j + 1].index
        x0 = xtrain[xtrain.index.isin(idx0)]
        x1 = xtrain[xtrain.index.isin(idx1)]
        y0 = ytrain[ytrain.index.isin(idx0)]
        y1 = ytrain[ytrain.index.isin(idx1)]
        
        #lr = LR(C = 1)														
        #lr.fit( np.array(x0)[:,0].reshape( -1, 1 ), y0 )
        y_raw = np.array(x1)[:,0]
        #y_platt = lr.predict_proba(np.array(x1)[:,0].reshape(-1,1))[:,1]
        ir = IR( out_of_bounds = 'clip' )	
        ir.fit( np.array(x0)[:,0], y0 )
        y_iso = ir.transform((np.array(x1)[:,0]))
        yval[idx1] = y_iso
        
        print(log_loss(y1, y_raw))
        print(log_loss(y1, y_iso))
        print(log_loss(y1, 0.5 * (y_iso + y_raw)))
        print(log_loss(y1,y_iso) - log_loss(y1,y_raw))        
                
                
        storage_mat[j, 0] = log_loss(y1, y_raw)
        storage_mat[j, 1] = log_loss(y1, y_iso)
        storage_mat[j, 2] = log_loss(y1,0.5 * (y_raw +  y_iso))
        
    # populate the new prfull
    ir = IR( out_of_bounds = 'clip' )	
    ir.fit( np.array(xtrain)[:,0], ytrain )
    yfull = ir.transform((np.array(xtest)[:,0]))
    
    prval2 = pd.DataFrame({'XGB1234kb4c':yval,'ID':id_train, 'target': ytrain})
    prfull2 = pd.DataFrame({'XGB1234kb4c':yfull, 'ID': id_test})
    prval2.to_csv('../metafeatures/prval_XGB_20160227c_seed1234.csv', index = False, header = True)
    prfull2.to_csv('../metafeatures/prfull_XGB_20160227c_seed1234.csv', index = False, header = True)
    
    # create forecasts
    pr_ref = pd.DataFrame({'ID':id_test, 'PredictedProb': xtest.XGB1234kb4})
    pr_ref.to_csv('../submissions/prfull_XGBref.csv', index = False, header = True)
    pr_calib = pd.DataFrame({'ID':id_test, 'PredictedProb': prfull2.XGB1234kb4c})
    pr_calib.to_csv('../submissions/prfull_XGBcalib.csv', index = False, header = True)
