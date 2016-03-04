# coding: utf-8
__author__ = 'mpearmain'


import pandas as pd
import xgboost as xgb
import datetime
import os

# settings
projPath = os.getcwd()
dataset_version = "kb6099"
todate = datetime.datetime.now().strftime("%Y%m%d")    
no_bags = 5
    
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
    clf = xgb.XGBClassifier(n_estimators=1520,
                            nthread=-1,
                            max_depth=11,
                            min_child_weight = 12.326654624743421,
                            learning_rate= 0.013901402098648891,
                            subsample= 0.80499400546083566,
                            colsample_bytree= 0.77245797080355449,
                            gamma=0.00060856204654098059,                             
                            seed=k*100+22, 
                            silent=True)
                            
    clf.fit(xtrain, ytrain, eval_metric='logloss')
    preds = clf.predict_proba(xtest)[:,1]
    if type(pred_average) == bool:
        pred_average = preds.copy()/no_bags
    else:
        pred_average += preds/no_bags
    print 'Finished bag:', k

sample.PredictedProb = pred_average
todate = datetime.datetime.now().strftime("%Y%m%d")
sample.to_csv('./submissions/xgboost_data_'+dataset_version+'_'+str(no_bags)+'bag_'+todate+'.csv', index=False)
