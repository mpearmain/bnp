# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 15:25:30 2016

@author: konrad
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from itertools import product
import datetime
from sklearn.feature_selection import GenericUnivariateSelect, chi2, SelectKBest

if __name__ == '__main__':

    ## settings
    dataset_version = "lvl220160407"
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


    ## feature selection: univariate selection


    ## feature selection: chi2

    ## feature selection: SelectKBest

    ## feature selection: RFE
