# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import manifold
import datetime

if __name__ == '__main__':

    ## settings
    projPath = './'
    dataset_version = "mp1"
    model_type = "manifolds"
    seed_value = 123
    todate = datetime.datetime.now().strftime("%Y%m%d")
    	    
    ## data
    # read the training and test sets
    xtrain = pd.read_csv(projPath + 'input/xtrain_'+ dataset_version + '.csv')     
    id_train = xtrain.ID
    ytrain = xtrain.target
    xtrain.drop('ID', axis = 1, inplace = True)
    xtrain.drop('target', axis = 1, inplace = True)
    
    xtest = pd.read_csv(projPath + 'input/xtest_'+ dataset_version + '.csv')     
    id_test = xtest.ID
    xtest.drop('ID', axis = 1, inplace = True)
        
    # Join the datasets (train and test to make a single dataset to build the
    # manifold projections on.
    train_rows = xtrain.shape[0]
    mani_data = pd.concat([xtrain, xtest])
    del xtrain, xtest
    # Setup the projections to be a 2D shape.
    n_components = 2

    # Build the different projection type.
    tsne = manifold.TSNE(n_components=n_components,
                         init='pca',
                         random_state=seed_value)
    foo = tsne.fit_transform(mani_data)

    # save the files
    mvalid.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    mfull.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
