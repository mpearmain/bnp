# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 15:25:30 2016

@author: konrad
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, make_scorer
from itertools import product
import datetime
from sklearn.feature_selection import GenericUnivariateSelect, chi2, SelectKBest

if __name__ == '__main__':

    input_folder = 'input2'
    target_folder = 'input2'
    
    ## settings
    dataset_version = "20160416v2"
    seed_value = 789
    todate = datetime.datetime.now().strftime("%Y%m%d")
    	    
    ## data
    xtrain = pd.read_csv('../'+input_folder+'/xtrain_'+ dataset_version + '.csv')     
    id_train = xtrain.ID
    y = xtrain.target
    xtrain.drop('target', axis = 1, inplace = True)
    xtrain.drop('ID', axis = 1, inplace = True)

    xtest = pd.read_csv('../'+input_folder+'/xtest_'+ dataset_version + '.csv')     
    id_test = xtest.ID
    xtest.drop('ID', axis = 1, inplace = True)

    # work in progress
    ## feature selection: SelectKBest based on Chi2 statistics

    kvals = [25, 50, 100]
    for kk in kvals:        
        xtr = np.array(xtrain); ytr = y.values; xte = np.array(xtest) 
        selector = SelectKBest(chi2, k = kk).fit(xtr, ytr)
        xtr2 = pd.DataFrame(selector.transform(xtr))
        xte2 = pd.DataFrame(selector.transform(xte))
        xtr2.columns = ['x'+str(f) for f in range(0,xtr2.shape[1])]
        xte2.columns = ['x'+str(f) for f in range(0,xte2.shape[1])]
        xtr2['target'] = y; xtr2['ID'] = id_train; xte2['ID'] = id_test     
        newname = dataset_version[-4:] + 'k' + str(kk)
        
        xtr2.to_csv('../'+target_folder+'/xtrain_' + newname + '.csv', index = False, header = True)
        xte2.to_csv('../'+target_folder+'/xtest_' + newname + '.csv', index = False, header = True)
        print(newname)
        
