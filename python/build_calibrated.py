# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 08:32:33 2016

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
from sklearn.isotonic import IsotonicRegression as IR
from os import listdir
import datetime


if __name__ == "__main__":

    # xtrain_lvl320160315.csv
    c_val = 2
    dataset_version = "lvl220160406combo"

    # read data
    xtrain = pd.read_csv('../input2/xtrain_' + dataset_version + '.csv')
    xtest = pd.read_csv('../input2/xtest_' + dataset_version + '.csv')
    
    # Read xfolds only need the ID and fold 5.
    print("Reading cross folds")
    xfolds = pd.read_csv('../input/xfolds.csv', usecols=['ID','fold5'])
    nfolds = len(np.unique(xfolds.fold5))
    
    # separate non-prediction columns
    id_train = xtrain.ID;  y = xtrain.target; id_test = xtest.ID
    xtrain.drop('ID', axis = 1, inplace = True)
    xtrain.drop('target', axis = 1, inplace = True)
    xtest.drop('ID', axis = 1, inplace = True)
    
    xtrain2 = np.zeros(xtrain.shape);    xtest2 = np.zeros(xtest.shape)
    
    ## data calibration
    for wfold in range(0, xtrain.shape[1]):
        print("----------------------------")
        print("column " + str(wfold) + ": " + xtrain.columns[wfold])
        # storage structures
        storage_mat = np.zeros((nfolds, 13))
        ymat_valid = np.zeros((xtrain.shape[0],13))
        ymat_full = np.zeros((xtest.shape[0],13))
        
        # create validated calibrationss
        for j in range(0,nfolds):
           # print("fold: " + str(j))
           idx0 = xfolds[xfolds.fold5 != j + 1].index
           idx1 = xfolds[xfolds.fold5 == j + 1].index
           x0 = xtrain[xtrain.index.isin(idx0)]
           x1 = xtrain[xtrain.index.isin(idx1)]
           y0 = y[y.index.isin(idx0)]
           y1 = y[y.index.isin(idx1)]
       
           y_raw = np.array(x1)[:,wfold]
           storage_mat[j,0] = log_loss(y1, y_raw)
           ymat_valid[idx1,0] = y_raw
           
           # fit an isotonic regression for iso scaling
           ir = IR( out_of_bounds = 'clip' )	
           ir.fit( np.array(x0)[:,wfold], y0 )
           y_iso = ir.transform((np.array(x1)[:,0]))           
           storage_mat[j,1] = log_loss(y1, y_iso)        
           ymat_valid[idx1,1] = y_iso
           storage_mat[j,7] = log_loss(y1, y_iso + y0.mean() - y_iso.mean())
           ymat_valid[idx1,7] = y_iso + y0.mean() - y_iso.mean()

            # fit a logistic regression for Platt scaling           
           lr = LR(C = c_val)														
           lr.fit( np.array(x0)[:,0].reshape( -1, 1 ), y0 )
           y_platt = lr.predict_proba(np.array(x1)[:,0].reshape(-1,1))[:,1]
           storage_mat[j,2] = log_loss(y1, y_platt)
           ymat_valid[idx1,2] = y_platt
           storage_mat[j,8] = log_loss(y1, y_platt + y0.mean() - y_platt.mean())
           ymat_valid[idx1,8] = y_platt + y0.mean() - y_platt.mean()
           
           y_ri = 0.5 * (y_raw + y_iso)
           storage_mat[j,3] = log_loss(y1, y_ri)
           ymat_valid[idx1,3] = y_ri
           # version with "normalized" mean
           storage_mat[j,9] = log_loss(y1, y_ri + y0.mean() - y_ri.mean())
           ymat_valid[idx1,9] = y_ri + y0.mean() - y_ri.mean()
           
           y_rp =  0.5 * (y_raw + y_platt)
           storage_mat[j,4] = log_loss(y1, y_rp)
           ymat_valid[idx1,4] = y_rp
           storage_mat[j,10] = log_loss(y1, y_rp + y0.mean() - y_rp.mean())
           ymat_valid[idx1,10] = y_rp + y0.mean() - y_rp.mean()
           
           y_ip = 0.5 * (y_iso + y_platt)
           storage_mat[j,5] = log_loss(y1, y_ip )
           ymat_valid[idx1,5] = y_ip
           storage_mat[j,11] = log_loss(y1, y_ip + y0.mean() - y_ip.mean())
           ymat_valid[idx1,11] = y_ip + y0.mean() - y_ip.mean()
           
           y_rpi = 0.5 * (y_raw+ 0.5 * (y_platt + y_iso))
           storage_mat[j,6] = log_loss(y1, y_rpi)
           ymat_valid[idx1,6] = y_rpi
           storage_mat[j,12] = log_loss(y1, y_rpi + y0.mean() - y_rpi.mean())
           ymat_valid[idx1,12] = y_rpi + y0.mean() - y_rpi.mean()
           
        # create full calibrations
        y_raw = np.array(xtest)[:, wfold]     
        ymat_full[:,0] = y_raw   
  
        ir = IR( out_of_bounds = 'clip' )	
        ir.fit( np.array(xtrain)[:,wfold], y )
        y_iso = ir.transform((np.array(xtest)[:,wfold]))
        ymat_full[:,1] = y_iso
        ymat_full[:,7] = y_iso + y.mean() - y_iso.mean()
        
        lr = LR(C = c_val)														
        lr.fit( np.array(xtrain)[:,wfold].reshape( -1, 1 ), y )
        y_platt= lr.predict_proba(np.array(xtest)[:,wfold].reshape(-1,1))[:,1]
        ymat_full[:,2] = y_platt
        ymat_full[:,8] = y_platt + y.mean() - y_platt.mean()
           
        y_ri  = 0.5 * (y_raw + y_iso)  
        ymat_full[:,3] = y_ri
        ymat_full[:,9] = y_ri + y.mean() - y_ri.mean()
        
        y_rp =  0.5 * (y_raw + y_platt)
        ymat_full[:,4] = y_rp
        ymat_full[:,10] = y_rp + y.mean() - y_rp.mean()
        
        y_ip = 0.5 * (y_iso + y_platt)
        ymat_full[:,5] = y_ip
        ymat_full[:,11] = y_ip + y.mean() - y_ip.mean()
           
        y_rpi = 0.5 * (y_raw+ 0.5 * (y_platt + y_iso))
        ymat_full[:,6] = y_rpi
        ymat_full[:,12] = y_rpi + y.mean() - y_rpi.mean()

        # pick the best performing one - that's the one we propagate to xvalid2/xfull2
        wbest = np.argmin(storage_mat.mean(axis = 0))
        xtrain2[:,wfold] = ymat_valid[:,wbest]
        xtest2[:,wfold] = ymat_full[:,wbest]

        a1 = log_loss(y, np.array(xtrain)[:,wfold])
        a2 = log_loss(y, np.array(xtrain2)[:,wfold])
        print("old logloss: %.6f "%a1)
        print("new logloss: %.6f "%a2)
        print("improvement: %.6f" %(a1 - a2))
    

    
        
    ## store calibrated version
    xtrain2 = pd.DataFrame(xtrain2); xtrain2.columns = xtrain.columns
    xtest2 = pd.DataFrame(xtest2); xtest2.columns = xtest.columns
    
    ## add summary statistics: mean, median, min, max
    xtrain2['xmin'] = xtrain2.min(axis = 1)
    xtest2['xmin'] = xtest2.min(axis = 1)
    xtrain2['xmax'] = xtrain2.max(axis = 1)
    xtest2['xmax'] = xtest2.max(axis = 1)
    xtrain2['xmedian'] = xtrain2.median(axis = 1)
    xtest2['xmedian'] = xtest2.median(axis = 1)
    
    xtrain2['target'] = y; xtrain2['ID'] = id_train
    xtest2['ID'] = id_test
    
    
    xtrain2.to_csv('../input2/xtrain_c'+str(c_val)+'d'+dataset_version+'.csv', index = False, header = True)
    xtest2.to_csv('../input2/xtest_c'+str(c_val)+'d'+dataset_version+'.csv', index = False, header = True)
    
        
