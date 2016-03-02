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
    projPath = os.getcwd()
    print(projPath)
    
    # Read xfolds only need the ID and fold 5.
    print("Reading Cross folds")
    xfolds = pd.read_csv('./input/xfolds.csv', usecols=['ID','fold5'])
    nfolds = len(np.unique(xfolds.fold5))
    
    # read the metas to calibrate
    print("Reading Train set")
    train = pd.read_csv('./metafeatures/prval_mars_20160228_datakb4_seed1901.csv')
    id_train = train.ID
    ytrain = train.target
    
    for j in range(0,nfolds):
        print("fold: " + str(j))
        idx0 = xfolds[xfolds.fold5 != j + 1].index
        idx1 = xfolds[xfolds.fold5 == j + 1].index
        x0 = train[train.index.isin(idx0)]
        x1 = train[train.index.isin(idx1)]
        y0 = ytrain[ytrain.index.isin(idx0)]
        y1 = ytrain[ytrain.index.isin(idx1)]
        
        lr = LR(C = 1)														
        lr.fit( np.array(x0)[:,0].reshape( -1, 1 ), y0 )
        y_raw = np.array(x1)[:,0]
        y_platt = lr.predict_proba(np.array(x1)[:,0].reshape(-1,1))[:,1]
        ir = IR( out_of_bounds = 'clip' )	# out_of_bounds param needs scikit-learn >= 0.15
        ir.fit( np.array(x0)[:,0], y0 )
        y_iso = ir.transform((np.array(x1)[:,0]))
        print(log_loss(y1, y_raw))
        print(log_loss(y1, y_platt))
        print(log_loss(y1, y_iso))
                
                