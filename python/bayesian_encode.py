import sys
import pandas as pd
import numpy as np
import operator
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn import ensemble
from sklearn.metrics import roc_auc_score,log_loss
import xgboost as xgb

def getCountVar(compute_df, count_df, var_name):
        grouped_df = count_df.groupby(var_name)
        count_dict = {}
        for name, group in grouped_df:
                count_dict[name] = group.shape[0]

        count_list = []
        for index, row in compute_df.iterrows():
                name = row[var_name]
                count_list.append(count_dict.get(name, 0))
        return count_list

def create_feature_map(features):
        outfile = open('xgb.fmap', 'w')
        for i, feat in enumerate(features):
                outfile.write('{0}\t{1}\tq\n'.format(i,feat))
        outfile.close()

def getDVEncodeVar(compute_df, purchase_df, var_name):
        grouped_df = purchase_df.groupby(var_name)
        zero_dict = {}
        one_dict = {}
        two_dict = {}
        for name, group in grouped_df:
                #if group.shape[0] > 1:
                        zero_dict[name] = (np.mean(np.array(group["target"]==1).astype('int'))) #* np.mean(np.array(group["fault_severity"]==0).astype('int'))
                        #one_dict[name] = (np.sum(np.array(group["fault_severity"]==1).astype('int'))) #* np.mean(np.array(group["fault_severity"]==1).astype('int'))
                        #two_dict[name] = (np.sum(np.array(group["fault_severity"]==2).astype('int'))) #* np.mean(np.array(group["fault_severity"]==2).astype('int'))


        zero_list = []
        one_list = []
        two_list = []
        for index, row in compute_df.iterrows():
                name = row[var_name]
                zero_list.append(zero_dict.get(name,-1))
                #one_list.append(one_dict.get(name,-1))
                #two_list.append(two_dict.get(name,-1))

        return zero_list


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None):
	params = {}
	params["objective"] = "binary:logistic"
	params['eval_metric'] = 'logloss'
	params["eta"] = 0.02
	params["min_child_weight"] = 1 
	params["subsample"] = 0.85
	params["colsample_bytree"] = 0.8
	params["silent"] = 1
	params["max_depth"] = 10
	params["seed"] = 12345
	#params["gamma"] = 0.5
	num_rounds = 10000

	plst = list(params.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	
	if test_y is not None:
	        xgtest = xgb.DMatrix(test_X, label=test_y)
	        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
	        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
	else:
		xgtest = xgb.DMatrix(test_X)
		model = xgb.train(plst, xgtrain, num_rounds)
	
	if feature_names:
                        create_feature_map(feature_names)
                        importance = model.get_fscore(fmap='xgb.fmap')
                        importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
                        imp_df = pd.DataFrame(importance, columns=['feature','fscore'])
                        imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()
                        imp_df.to_csv("imp_feat.txt", index=False)
	
	pred_test_y = model.predict(xgtest)
	
	if test_y is not None:
	        loss = log_loss(test_y, pred_test_y)
		print loss
	
	return pred_test_y, loss  

if __name__ == "__main__":
	import datetime
	start_time = datetime.datetime.now()
	print "Start time : ", start_time

	print "Reading files.."
	train = pd.read_csv('../Data/train.csv')
	test = pd.read_csv('../Data/test.csv')
	print train.shape, test.shape

	print "Filling NA.."
	train = train.fillna(-1)
	test = test.fillna(-1)

	print "Label encoding.."
	cat_columns = []
	for f in train.columns:
		if train[f].dtype=='object':
			print(f), len(np.unique(train[f].values))
			#if f != 'v22':
			cat_columns.append(f)
	        	lbl = preprocessing.LabelEncoder()
	        	lbl.fit(list(train[f].values) + list(test[f].values))
	        	train[f] = lbl.transform(list(train[f].values))
	        	test[f] = lbl.transform(list(test[f].values))
			new_train = pd.concat([ train[['v1',f]], test[['v1',f]] ])
			train["CountVar_"+str(f)] = getCountVar(train[['v1',f]], new_train[['v1', f]], f)
                test["CountVar_"+str(f)] = getCountVar(test[['v1',f]], new_train[['v1',f]], f)


	print "Encoding train...."
	for f in cat_columns:
		print f
		val_list = np.zeros(train.shape[0])
		folds_array = np.array( pd.read_csv("../Data/cv_folds.csv")["CVFold"] )
        	for fold_index in xrange(1,6):
                	dev_index = np.where(folds_array != fold_index)[0]
                	val_index = np.where(folds_array == fold_index)[0]
			new_train = train[["v1", f, "target"]]
			dev, val = new_train.iloc[dev_index,:], new_train.iloc[val_index,:]
			enc_list =  np.array( getDVEncodeVar(val[["v1", f]], dev[["v1", f, "target"]], f)  )
			val_list[val_index] = enc_list
		train["DVEncode_"+str(f)] =  val_list
		#print np.unique(train["DVEncode_"+str(f)])

	print "Encoding test.."
	for f in cat_columns:
		print f
		test["DVEncode_"+str(f)] = getDVEncodeVar(test[["v1", f]], train[["v1", f, "target"]], f)

	print "Two way encoding.."
	new_var_list = []
	var1_cols = ["v22", "v56", "v125"]
	var2_cols = [col for col in cat_columns if col not in var1_cols]
	for ind, var1 in enumerate(var1_cols):
                #rem_cols = cat_columns[ind+1:]
                #if var1 in "v30":
                #       break
                for var2 in var2_cols:
                        print var1, var2
                        new_var = var1+"_"+var2
                        train[new_var] = train[[var1,var2]].apply(lambda row: str(row[0])+"_"+str(row[1]),axis=1)
                        test[new_var] = test[[var1,var2]].apply(lambda row: str(row[0])+"_"+str(row[1]),axis=1)
			new_train = pd.concat([ train[['v1',new_var]], test[['v1',new_var]] ])
                        test["Count_"+new_var] = getCountVar(test[['v1',new_var]], new_train[['v1', new_var]], new_var)
                        train["Count_"+new_var] = getCountVar(train[['v1',new_var]], new_train[['v1', new_var]], new_var)
			new_var_list.append(new_var)

	print "Train.."
	for f in new_var_list:
                print f
                val_list = np.zeros(train.shape[0])
		folds_array = np.array( pd.read_csv("../Data/cv_folds.csv")["CVFold"] )
                for fold_index in xrange(1,6):
                        dev_index = np.where(folds_array != fold_index)[0]
                        val_index = np.where(folds_array == fold_index)[0]
			new_train = train[["v1", f, "target"]]
                        dev, val = new_train.iloc[dev_index,:], new_train.iloc[val_index,:]
                        enc_list =  np.array( getDVEncodeVar(val[["v1", f]], dev[["v1", f, "target"]], f)  )
                        val_list[val_index] = enc_list
                train["DVEncode_"+str(f)] =  val_list

	print "Test.."
	for f in new_var_list:
		print f
		test["DVEncode_"+str(f)] = getDVEncodeVar(test[["v1", f]], train[["v1", f, "target"]], f)

	train = train.drop(new_var_list, axis=1)
	test = test.drop(new_var_list, axis=1)
	train.to_csv("train_dvencode.csv", index=False)
	test.to_csv("test_dvencode.csv", index=False)
		
	end_time = datetime.datetime.now()
	print "End time : ",end_time

	print end_time - start_time
