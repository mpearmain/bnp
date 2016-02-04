# coding: utf-8
__author__ = 'mpearmain'


import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score as auc
import datetime

# settings
projPath = './'
dataset_version = "ensemble_base"
todate = datetime.datetime.now().strftime("%Y%m%d")    
no_bags = 5
    
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

# Get rid of incorrect names for xgboost (scv-rbf) cannot handle '-'
xtrain = xtrain.rename(columns=lambda x: x.replace('-', ''))
xtest = xtest.rename(columns=lambda x: x.replace('-', ''))

sample = pd.read_csv(projPath + 'input/sample_submission.csv')

pred_average = True

for k in range(no_bags):
    print 'Building bag:', k
    clf = xgb.XGBClassifier(n_estimators=350,
                            nthread=-1,
                            max_depth=8,
                            min_child_weight = 8.4966878568341748,
                            learning_rate= 0.017668218154585136,
                            silent=True,
                            subsample=0.85005533637305219,
                            colsample_bytree=0.80496791490053055,
                            gamma=0.00040708117094670357,
                            seed=k*100+22)
     
                                        
    clf.fit(xtrain, ytrain)
    preds = clf.predict_proba(xtest)[:,1]
    if type(pred_average) == bool:
        pred_average = preds.copy()/no_bags
    else:
        pred_average += preds/no_bags


sample.target = pred_average
todate = datetime.datetime.now().strftime("%Y%m%d")
sample.to_csv(projPath + 'submissions/xgb_meta_data'+dataset_version+'_'+str(no_bags)+'bag_'+todate+'.csv', index=False)