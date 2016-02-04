# coding: utf-8
__author__ = 'mpearmain'


import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import datetime

# settings
projPath = './'
dataset_version = "ensemble_base"
todate = datetime.datetime.now().strftime("%Y%m%d")    
no_bags = 5
    
## data
# read the training and test sets
xtrain = pd.read_csv(projPath + 'input/xvalid_'+ dataset_version + '.csv')
id_train = xtrain.ID
ytrain = xtrain.target
xtrain.drop('ID', axis = 1, inplace = True)
xtrain.drop('target', axis = 1, inplace = True)

xtest = pd.read_csv(projPath + 'input/xfull_'+ dataset_version + '.csv')
id_test = xtest.ID
xtest.drop('ID', axis = 1, inplace = True)

sample = pd.read_csv(projPath + 'input/sample_submission.csv')

pred_average = True

for k in range(no_bags):
    print 'Building bag:', k
    clf = xgb.XGBClassifier(n_estimators=573,
                            nthread=-1,
                            max_depth=5,
                            min_child_weight = 4.686493440588162,
                            learning_rate= 0.0085483037718426733,
                            subsample=0.830,
                            colsample_bytree=0.82,
                            gamma=0.00020709838670210425,
                            seed=k*100+22,
                            silent=True)
    clf.fit(xtrain, ytrain, eval_metric='logloss')
    clf_preds = clf.predict_proba(xtest)[:,1]
    # Now calibrate
    isopreds = clf.predict_proba(xtrain)[:,1]
    iso = IsotonicRegression()
    iso.fit(isopreds, ytrain)
    preds = iso.predict(clf_preds)

    if type(pred_average) == bool:
        pred_average = preds.copy()/no_bags
    else:
        pred_average += preds/no_bags

sample.PredictedProb = pred_average
todate = datetime.datetime.now().strftime("%Y%m%d")
sample.to_csv(projPath + 'submissions/xgb_meta_data'+dataset_version+'_'+str(no_bags)+'bag_'+todate+'.csv', index=False)