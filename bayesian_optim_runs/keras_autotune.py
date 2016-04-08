from __future__ import division
from __future__ import print_function

__author__ = 'michael.pearmain'

import pandas as pd
import numpy as np
import datetime
from bayes_opt import BayesianOptimization
from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Dense, regularizers
from keras.utils import np_utils
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def kerascv(dense1, dropout1, dense2, dropout2, epochs):
    # setup bagging classifier
    pred_sum = 0
    for k in range(3):
        model = createModel(dense1=int(dense1), dropout1=dropout1, dense2=int(dense2), dropout2=dropout2)
        model.fit(x0, y0, nb_epoch=int(epochs), batch_size=256, verbose=0)

        preds = model.predict_proba(x1)[:,1]
        pred_sum += preds
        pred_average = pred_sum / (k+1)
        del model

    ll = -log_loss(y1[:,1],pred_average)
    return ll


def createModel(dense1, dropout1, dense2, dropout2):
    model = Sequential()
    model.add(Dense(dense1, input_shape=(dims,), init='he_uniform', W_regularizer=regularizers.l1(0.0005)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout1))# input dropout
    model.add(Dense(dense2, init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(dropout2))
    model.add(Dense(2, init='he_uniform'))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="adagrad")
    return model

def getDummy(df,col):
        category_values=df[col].unique()
        data=[[0 for i in range(len(category_values))] for i in range(len(df))]
        dic_category=dict()
        for i,val in enumerate(list(category_values)):
            dic_category[str(val)]=i
       # print dic_category
        for i in range(len(df)):
            data[i][dic_category[str(df[col][i])]]=1

        data=np.array(data)
        for i,val in enumerate(list(category_values)):
            df.loc[:,"_".join([col,str(val)])]=data[:,i]

        return df

if __name__ == "__main__":
    ## settings
    dataset_version = "dvencode_3level4"
    nbag = 5
    seed_value = 1543
    todate = datetime.datetime.now().strftime("%Y%m%d")
    np.random.seed(seed_value)
    need_normalise=True
    need_categorical=False

    xtrain = pd.read_csv('../input/xtrain_'+ dataset_version + '.csv')
    id_train = xtrain.ID
    y_train_target = xtrain.target
    ytrain = xtrain.target
    xtrain.drop('ID', axis = 1, inplace = True)
    xtrain.drop('target', axis = 1, inplace = True)

    test = pd.read_csv('../input/xtest_'+ dataset_version + '.csv')
    id_test = test.ID
    test.drop('ID', axis = 1, inplace = True)

    encoder = LabelEncoder()
    ytrain = encoder.fit_transform(ytrain).astype(np.int32)
    ytrain = np_utils.to_categorical(ytrain)

    print ("processsing finished")
    xtrain = np.array(xtrain)
    xtrain = xtrain.astype(np.float32)
    test = np.array(test)
    test = test.astype(np.float32)
    if need_normalise:
        scaler = StandardScaler().fit(xtrain)
        xtrain = scaler.transform(xtrain)
        test = scaler.transform(test)

    # folds
    xfolds = pd.read_csv('../input/xfolds.csv')
    # work with 5-fold split
    fold_index = np.array(xfolds.valid)

    # work with validation split
    idx0 = np.where(fold_index != 1)
    idx1 = np.where(fold_index == 1)
    x0 = np.array(xtrain)[idx0,:][0]
    x1 = np.array(xtrain)[idx1,:][0]
    y0 = np.array(ytrain)[idx0]
    y1 = np.array(ytrain)[idx1]

    nb_classes = 2
    dims = xtrain.shape[1]
    print(dims, 'dims')

    kerasBO = BayesianOptimization(kerascv,
                                   {'dense1': (int(0.1 * xtrain.shape[1]), int(2 * xtrain.shape[1])),
                                    'dropout1': (0.2, 0.7),
                                    'dense2': (int(0.1 * xtrain.shape[1]), int(2 * xtrain.shape[1])),
                                    'dropout2': (0.2, 0.7),
                                    'epochs': (int(30), int(200))
                                    })
    kerasBO.maximize(init_points=5, n_iter=20)
    print('-' * 53)

    print('Final Results')
    print('Extra Trees: %f' % kerasBO.res['max']['max_val'])
    print(kerasBO.res['max']['max_params'])
