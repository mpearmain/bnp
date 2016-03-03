# coding: utf-8
__author__ = 'mpearmain'


import pandas as pd
import xgboost as xgb
import datetime
import os

# settings
projPath = os.getcwd()
dataset_version = "ensemble_base"
todate = datetime.datetime.now().strftime("%Y%m%d")    
no_bags = 3
    
## data
# read the training and test sets
xtrain = pd.read_csv('./input/xvalid_'+ dataset_version + '.csv')
id_train = xtrain.ID
ytrain = xtrain.target
xtrain.drop('ID', axis = 1, inplace = True)
xtrain.drop('target', axis = 1, inplace = True)

xtest = pd.read_csv('./input/xfull_'+ dataset_version + '.csv')
id_test = xtest.ID
xtest.drop('ID', axis = 1, inplace = True)

sample = pd.read_csv('./input/sample_submission.csv')

pred_average = True

for k in range(no_bags):
    print 'Building bag:', k
    clf = xgb.XGBClassifier(n_estimators=648,
                            nthread=-1,
                            max_depth=9,
                            min_child_weight = 3.5928,
                            learning_rate= 0.0077,
                            subsample= 0.7296,
                            colsample_bytree= 0.7829,
                            gamma=0.0061,
                            seed=k*100+22,
                            silent=True)

    clf.fit(xtrain, ytrain, eval_metric='logloss')
    preds = clf.predict_proba(xtest)[:,1]
    if type(pred_average) == bool:
        pred_average = preds.copy()/no_bags
    else:
        pred_average += preds/no_bags

sample.PredictedProb = pred_average
todate = datetime.datetime.now().strftime("%Y%m%d")
sample.to_csv('./submissions/xgboost_data_'+dataset_version+'_'+str(no_bags)+'bag_'+todate+'.csv', index=False)
