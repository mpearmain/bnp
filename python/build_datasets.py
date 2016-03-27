# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:58:51 2016

@author: konrad
based on: https://www.kaggle.com/scirpus/bnp-paribas-cardif-claims-management/benouilli-naive-bayes/code
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.naive_bayes import BernoulliNB
from itertools import combinations
from sklearn.decomposition import TruncatedSVD

def buildKB15():
    ## data
    # read the training/test data  
    print('Importing Data')
    xtrain = pd.read_csv('../input/train.csv')
    xtest = pd.read_csv('../input/test.csv')
    
    xtrain.fillna(-1, inplace=True)
    xtest.fillna(-1, inplace=True)
    
    # separate 
    id_train = xtrain.ID; xtrain.drop('ID', axis = 1, inplace = True)
    ytrain = xtrain.target; xtrain.drop('target', axis = 1, inplace = True)
    id_test = xtest.ID; xtest.drop('ID', axis = 1, inplace = True)
    
    # drop v22 - categorical with 18211 possible values 
    xtrain.drop('v22', axis = 1, inplace = True)
    xtest.drop('v22', axis = 1, inplace = True)
    
    # folds for cv   
    xfolds = pd.read_csv('../input/xfolds.csv')
    fold_index = xfolds.fold5
    fold_index = np.array(fold_index) - 1
    n_folds = len(np.unique(fold_index))

    ## processing
    # identify columns classes
    categorical_cols = [f for f in xtrain.columns if xtrain[f].dtype not in ['float64', 'int64']]
    numerical_cols = [f for f in xtrain.columns if xtrain[f].dtype in ['float64']]
    
    # number of unique values
    # headcounts = [len(np.unique(xtrain[f])) for f in categorical_cols]
    
    # convert all categoricals: expand into binary indicators, use as features 
    # fed into NaiveBayes, drop the original
    for col in categorical_cols:
        print(col)
        newname = 'nb_' + col
        # transform the joint set into dummies 
        xloc = pd.concat((xtrain[col],xtest[col]), axis = 0, ignore_index = True)
        xloc = pd.get_dummies(xloc)
        # separate back into training and test
        xtr = xloc.ix[range(0,xtrain.shape[0])]
        xte = xloc.ix[range(xtrain.shape[0], xloc.shape[0])]
        # storage vector for the new features (train and test)
        newvar_train = np.zeros((xtrain.shape[0]))        
        # build a stacked version along the training set
        for j in range(0,n_folds):
            idx0 = np.where(fold_index != j)
            idx1 = np.where(fold_index == j)
            x0 = np.array(xtr)[idx0,:][0]; x1 = np.array(xtr)[idx1,:][0]
            y0 = np.array(ytrain)[idx0]; y1 = np.array(ytrain)[idx1]
            nb = BernoulliNB()
            nb.fit(x0,y0)            
            newvar_train[idx1] = nb.predict_proba(x1)[:,1]
            print(log_loss(y1, newvar_train[idx1]))
        # build a stacked version along the test set
        nb.fit(xtr, ytrain)
        newvar_test = nb.predict_proba(xte)[:,1]
        # park into training and test sets
        xtrain[newname] = newvar_train
        xtest[newname] = newvar_test
        xtrain.drop(col, axis = 1, inplace = True)
        xtest.drop(col, axis = 1, inplace = True)
                
         
                              
    ## store the results
    # add indices etc
    xtrain = pd.DataFrame(xtrain)
    xtrain['ID'] = id_train
    xtrain['target'] = ytrain
#
    xtest = pd.DataFrame(xtest)
    xtest['ID'] = id_test
#
#
#    # save the files
    xtrain.to_csv('../input/xtrain_kb15.csv', index = False, header = True)
    xtest.to_csv('../input/xtest_kb15.csv', index = False, header = True)
    
    return
    
    
def buildKB16(n_comp = 200, seed_value = 123):
    ## data
    # read the training/test data  
    print('Importing Data')
    xtrain = pd.read_csv('../input/xtrain_kb6099.csv')
    xtest = pd.read_csv('../input/xtest_kb6099.csv')
    
    # separate 
    id_train = xtrain.ID; xtrain.drop('ID', axis = 1, inplace = True)
    ytrain = xtrain.target; xtrain.drop('target', axis = 1, inplace = True)
    id_test = xtest.ID; xtest.drop('ID', axis = 1, inplace = True)
    
    # fit SVD
    svd = TruncatedSVD(n_components = n_comp,n_iter=5, random_state= seed_value)
    svd.fit(xtrain)
    xtrain = svd.transform(xtrain)
    xtest = svd.transform(xtest)
    xtrain = pd.DataFrame(xtrain)
    xtest = pd.DataFrame(xtest)
    
    ## store the results
    # add indices etc
    xtrain = pd.DataFrame(xtrain)
    xtrain['ID'] = id_train
    xtrain['target'] = ytrain
#
    xtest = pd.DataFrame(xtest)
    xtest['ID'] = id_test
#
#
#    # save the files
    xtrain.to_csv('../input/xtrain_kb16c'+str(n_comp)+'.csv', index = False, header = True)
    xtest.to_csv('../input/xtest_kb16c'+str(n_comp)+'.csv', index = False, header = True)
    
    return
    
if __name__ == "__main__":
    
    # buildKB15()
    buildKB16(n_comp = 200, seed_value = 12)
    buildKB16(n_comp = 300, seed_value = 12)