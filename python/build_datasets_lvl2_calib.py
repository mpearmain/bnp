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
from sklearn.isotonic import IsotonicRegression as IR
from os import listdir
import datetime

if __name__ == "__main__":


    todate = datetime.datetime.now().strftime("%Y%m%d") 
    ## data construction
    # list the groups 
    target_path = '../metafeatures/'
    xlist_val = [f for f in listdir(target_path) if 'prval' in f]
    xlist_full = [f for f in listdir(target_path) if 'prfull' in f]

    # aggregate the validation set
    ii = 0
    mod_class = xlist_val[ii].split('_')[1]
    xvalid = pd.read_csv(target_path + xlist_val[ii])
    mod_cols = [f for f in range(xvalid.shape[1]) if mod_class in xvalid.columns[f]]
    new_names = [mod_class + str(ff) for ff in range(len(mod_cols))]
    xcols = xvalid.columns.values
    for ff in range(len(mod_cols)):
        xcols[mod_cols[ff]] = new_names[ff]+str(ii)
    xvalid.columns = xcols
        
    for ii in range(1,len(xlist_val)):
        mod_class = xlist_val[ii].split('_')[1]
        xval = pd.read_csv(target_path + xlist_val[ii])
        mod_cols = [f for f in range(xval.shape[1]) if mod_class in xval.columns[f]]
        new_names = [mod_class + str(ff) for ff in range(len(mod_cols))]
        xcols = xval.columns.values
        for ff in range(len(mod_cols)):
            xcols[mod_cols[ff]] = new_names[ff]+str(ii)
        xval.columns = xcols
        xvalid = pd.merge(xvalid, xval)
        print(xvalid.shape)
        
    # aggregate the test set
    ii = 0
    mod_class = xlist_full[ii].split('_')[1]
    xfull = pd.read_csv(target_path + xlist_full[ii])
    mod_cols = [f for f in range(xfull.shape[1]) if mod_class in xfull.columns[f]]
    new_names = [mod_class + str(ff) for ff in range(len(mod_cols))]
    xcols = xfull.columns.values
    for ff in range(len(mod_cols)):
        xcols[mod_cols[ff]] = new_names[ff]+str(ii)
    xfull.columns = xcols
        
    for ii in range(1,len(xlist_val)):
        mod_class = xlist_full[ii].split('_')[1]
        xval = pd.read_csv(target_path + xlist_full[ii])
        mod_cols = [f for f in range(xval.shape[1]) if mod_class in xval.columns[f]]
        new_names = [mod_class + str(ff) for ff in range(len(mod_cols))]
        xcols = xval.columns.values
        for ff in range(len(mod_cols)):
            xcols[mod_cols[ff]] = new_names[ff]+str(ii)
        xval.columns = xcols
        xfull = pd.merge(xfull, xval)
        print(xfull.shape)       

    id_train = xvalid.ID;  y = xvalid.target; id_test = xfull.ID
    xvalid.drop('ID', axis = 1, inplace = True)
    xvalid.drop('target', axis = 1, inplace = True)
    xfull.drop('ID', axis = 1, inplace = True)
    
    # Read xfolds only need the ID and fold 5.
    print("Reading cross folds")
    xfolds = pd.read_csv('../input/xfolds.csv', usecols=['ID','fold5'])
    nfolds = len(np.unique(xfolds.fold5))

    # new storage matrices for calibrated versions
    xvalid2 = np.zeros(xvalid.shape);    xfull2 = np.zeros(xfull.shape)
    
    ## data calibration
    for wfold in range(0, xvalid.shape[1]):
        print("column: " + str(wfold))
        # storage structures
        storage_mat = np.zeros((nfolds, 5))
        ymat_valid = np.zeros((xvalid.shape[0],5))
        ymat_full = np.zeros((xfull.shape[0],5))
        
        # create validated calibrationss
        for j in range(0,nfolds):
           print("fold: " + str(j))
           idx0 = xfolds[xfolds.fold5 != j + 1].index
           idx1 = xfolds[xfolds.fold5 == j + 1].index
           x0 = xvalid[xvalid.index.isin(idx0)]
           x1 = xvalid[xvalid.index.isin(idx1)]
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

            # fit a logistic regression for Platt scaling           
           lr = LR(C = 1)														
           lr.fit( np.array(x0)[:,0].reshape( -1, 1 ), y0 )
           y_platt = lr.predict_proba(np.array(x1)[:,0].reshape(-1,1))[:,1]
           storage_mat[j,2] = log_loss(y1, y_platt)
           ymat_valid[idx1,2] = y_platt
           
           
           storage_mat[j,3] = log_loss(y1, 0.5 * (y_raw + y_iso))
           ymat_valid[idx1,3] = 0.5 * (y_raw + y_iso)
           
           storage_mat[j,4] = log_loss(y1, 0.5 * (y_raw + y_platt))
           ymat_valid[idx1,4] = 0.5 * (y_raw + y_platt)
           
           
        # create full calibrations
        ymat_full[:,0] = np.array(xfull)[:, wfold]        
  
        ir = IR( out_of_bounds = 'clip' )	
        ir.fit( np.array(xvalid)[:,wfold], y )
        ymat_full[:,1] = ir.transform((np.array(xfull)[:,wfold]))
        
        lr = LR(C = 1)														
        lr.fit( np.array(xvalid)[:,wfold].reshape( -1, 1 ), y )
        ymat_full[:,2] = lr.predict_proba(np.array(xfull)[:,wfold].reshape(-1,1))[:,1]
           
        ymat_full[:,3] = 0.5 * (ymat_full[:,0] + ymat_full[:,1])
        ymat_full[:,4] = 0.5 * (ymat_full[:,0] + ymat_full[:,2])
            
        # pick the best performing one - that's the one we propagate to xvalid2/xfull2
        wbest = np.argmin(storage_mat.mean(axis = 0))
        xvalid2[:,wfold] = ymat_valid[:,wbest]
        xfull2[:,wfold] = ymat_full[:,wbest]
                
    
    # SFSG # 
    ## data augmentation
    # xvalid2 - summary statistics
    xmed = np.median(xvalid2, axis = 1)
    xmin = np.min(xvalid2, axis = 1)
    xmax = np.max(xvalid2, axis = 1)
    xmean = np.mean(xvalid2, axis = 1)
    xq1 = np.percentile(xvalid2, q = 0.1, axis = 1)
    xq2 = np.percentile(xvalid2, q = 0.25, axis = 1)
    xq3 = np.percentile(xvalid2, q = 0.75, axis = 1)
    xq4 = np.percentile(xvalid2, q = 0.9, axis = 1)
    
    xvalid2 = pd.DataFrame(xvalid2); xvalid2.columns = xvalid.columns
    xvalid2['xmed'] = xmed; xvalid2['xmean'] = xmean
    xvalid2['xmin'] = xmin; xvalid2['xmax'] = xmax
    xvalid2['xq1'] = xq1; xvalid2['xq2'] = xq2
    xvalid2['xq3'] = xq3; xvalid2['xq4'] = xq4
    
    # prepare for dump
    xvalid2['ID'] = id_train
    xvalid2['target'] = y

    
    # xfull2 -  summary statistics 
    xmed = np.median(xfull2, axis = 1)
    xmin = np.min(xfull2, axis = 1)
    xmax = np.max(xfull2, axis = 1)
    xmean = np.mean(xfull2, axis = 1)
    xq1 = np.percentile(xfull2, q = 0.1, axis = 1)
    xq2 = np.percentile(xfull2, q = 0.25, axis = 1)
    xq3 = np.percentile(xfull2, q = 0.75, axis = 1)
    xq4 = np.percentile(xfull2, q = 0.9, axis = 1)
    
    xfull2 = pd.DataFrame(xfull2); xfull2.columns = xfull.columns
    xfull2['xmed'] = xmed; xfull2['xmean'] = xmean
    xfull2['xmin'] = xmin; xfull2['xmax'] = xmax
    xfull2['xq1'] = xq1; xfull2['xq2'] = xq2
    xfull2['xq3'] = xq3; xfull2['xq4'] = xq4
    
    xfull2['ID'] = id_test
           
    
    ## store the new datasets
    xvalid2.to_csv('../input/xtrain_lvlc2'+ todate + '.csv', index = False, header = True)
    xfull2.to_csv('../input/xtest_lvlc2'+ todate + '.csv', index = False, header = True)