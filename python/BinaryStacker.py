from __future__ import print_function
from __future__ import division

import numpy as np
from sklearn import cross_validation

class BinaryStackingClassifier():
    """
    To facilitate stacking of binary classifiers
    It only provides fit and predict_proba functions, and works with binary [0, 1] labels.

    :param base_classifiers: A list of binary classifiers with a fit and predict_proba method similar to that of sklearn
    :param xfolds: A cross folds pandas data set indicating the index and the fold number

    This stacking technique creates prediction dataset in one go
    """
    def __init__(self, base_classifiers, xfolds):
        self.base_classifiers = base_classifiers
        self.xfolds = xfolds


    def fit(self, X, y, **kwargs):
        stacking_train = np.full((np.shape(X)[0], len(self.base_classifiers)),np.nan)

        for model_no in range(len(self.base_classifiers)):
            cv = cross_validation.KFold(len(X), n_folds=self.n)
            for j, (traincv, testcv) in enumerate(cv):
                self.base_classifiers[model_no].fit(X[traincv, ], y[traincv], **kwargs)
                predicted_y_proba = self.base_classifiers[model_no].predict_proba(X[testcv,])[:, 1]
                stacking_train[testcv, model_no] = predicted_y_proba

            self.base_classifiers[model_no].fit(X, y)
        self.combiner.fit(stacking_train, y)

    def predict_proba(self, X):
        stacking_predict_data = np.full(
            (np.shape(X)[0], len(self.base_classifiers)),
            np.nan
        )
        for model_no in range(len(self.base_classifiers)):
            stacking_predict_data[:, model_no] = self.base_classifiers[model_no].predict_proba(X)[:, 1]
        return self.combiner.predict_proba(stacking_predict_data)[:, 1]