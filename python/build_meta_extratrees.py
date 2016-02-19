# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import

def extratreescv('max_depth'=max_depth,
                 'min_weight_fraction_leaf'=min_weight_fraction_leaf,
                 'max_leaf_nodes'=max_leaf_nodes,
                 ):

    clf = ExtraTreesClassifier(criterion='gini',
                               n_estimators=n_estimators,
                               max_depth=max_depth,
                               min_weight_fraction_leaf=min_weight_fraction_leaf,
                               max_leaf_nodes=max_leaf_nodes,
                               n_jobs= -1,
                               random_state= seed_values[i],
                               verbose=0)
    clf.fit(x0, y0)
    ll = -log_loss(y1, clf.predict_proba(x1)[:,1])
    return ll






if __name__ == '__main__':

    ## settings
    projPath = os.getcwd()
    dataset_version = ["mp1", "kb1", "kb2", "kb3", "kb4", "kb5095", "kb6095",
                       "kb6099"]
    model_type = "etrees"
    # Generate the same random sequences.
    np.random.seed(42)
    seed_values = np.random.randint(1, 1e10, len(dataset_version))
    todate = datetime.datetime.now().strftime("%Y%m%d")

    # Create the loops over the datasets.
    for i, dataset in enumerate(dataset_version):
        print 'Running Bayesian optimization for dataset:', dataset

        ## data
        # read the training and test sets
        xtrain = pd.read_csv(projPath + 'input/xtrain_'+ dataset + '.csv')
        id_train = xtrain.ID
        ytrain = xtrain.target
        xtrain.drop('ID', axis = 1, inplace = True)
        xtrain.drop('target', axis = 1, inplace = True)

        xtest = pd.read_csv(projPath + 'input/xtest_'+ dataset_version + '.csv')
        id_test = xtest.ID
        xtest.drop('ID', axis = 1, inplace = True)

        # folds
        xfolds = pd.read_csv(projPath + 'input/xfolds.csv')

        # We now work with the train and validation set for bayes opt.
        idx0 = xfolds[xfolds.valid == 0].index
        idx1 = xfolds[xfolds.valid == 1].index
        x0 = xtrain[xtrain.index.isin(idx0)]
        x1 = xtrain[xtrain.index.isin(idx1)]
        y0 = ytrain[ytrain.index.isin(idx0)]
        y1 = ytrain[ytrain.index.isin(idx1)]













        # work with 5-fold split
        fold_index = xfolds.fold5
        fold_index = np.array(fold_index) - 1
        n_folds = len(np.unique(fold_index))

        ## model
        # setup model instances
        model = ExtraTreesClassifier(criterion='gini',
                                     max_depth=None,
                                     min_weight_fraction_leaf=0.0,
                                     max_leaf_nodes=None,
                                     n_jobs= -1,
                                     random_state= seed_values[i],
                                     verbose=0)
        # parameter grids: LR + range of training subjects to subset to
        n_vals = [100, 500, 100]
        n_minleaf = [1,5, 25]
        n_minsplit = [2,10, 20]
        n_maxfeat = [0.05, 0.1, 0.22]
        param_grid = tuple([n_vals, n_minleaf, n_minsplit, n_maxfeat])
        param_grid = list(product(*param_grid))

        # storage structure for forecasts
        mvalid = np.zeros((xtrain.shape[0],len(param_grid)))
        mfull = np.zeros((xtest.shape[0],len(param_grid)))

        ## build 2nd level forecasts
        for i in range(len(param_grid)):
                print "processing parameter combo:", i, "of", len(param_grid)
                print "Combo:", param_grid[i]
                # configure model with j-th combo of parameters
                x = param_grid[i]
                model.n_estimators = x[0]
                model.min_samples_leaf = x[1]
                model.min_samples_split = x[2]
                model.max_features = x[3]

                # loop over folds
                for j in range(0,n_folds):
                    idx0 = np.where(fold_index != j)
                    idx1 = np.where(fold_index == j)
                    x0 = np.array(xtrain)[idx0,:][0]; x1 = np.array(xtrain)[idx1,:][0]
                    y0 = np.array(ytrain)[idx0]; y1 = np.array(ytrain)[idx1]
                    # fit the model on observations associated with subject whichSubject in this fold
                    model.fit(x0, y0)
                    mvalid[idx1,i] = model.predict_proba(x1)[:,1]
                    print 'Logloss on fold:', log_loss(y1, model.predict_proba(x1)[:,1])
                    print "finished fold:", j

                # fit on complete dataset
                model.fit(xtrain, ytrain)
                mfull[:,i] = model.predict_proba(xtest)[:,1]
                print "finished full prediction"

        # store the results
        # add indices etc
        mvalid = pd.DataFrame(mvalid)
        mvalid.columns = [model_type + str(i) for i in range(0, mvalid.shape[1])]
        mvalid['ID'] = id_train
        mvalid['target'] = ytrain

        mfull = pd.DataFrame(mfull)
        mfull.columns = [model_type + str(i) for i in range(0, mfull.shape[1])]
        mfull['ID'] = id_test


        # save the files
        mvalid.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
        mfull.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
