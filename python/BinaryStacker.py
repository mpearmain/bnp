from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd

class BinaryStackingClassifier():
    """
    To facilitate stacking of binary classifiers
    It only provides fit and predict_proba functions, and works with binary [0, 1] labels.

    :param base_classifiers: A list of binary classifiers with a fit and predict_proba method similar to that of sklearn
    :param xfolds: A cross folds pandas data frame indicating the identifier for fold selection (col1) and the fold
                   number (col2)
                   ID,fold5
                   3,   3
                   4,   5
                   5,   3
                   6,   3
                   8,   1
                   In this class the names do not matter it is positional.

                   ##############################
                   Fold number must start from 1.
                   ##############################

    :param evaluation: optional evaluation metric (y_true, y_score) to check metric at each fold.
                    expected use case might be evaluation=sklearn.Metrics.logLoss

    """
    def __init__(self, base_classifiers, xfolds, evaluation=None):
        self.base_classifiers = base_classifiers
        assert(xfolds.shape[1] == 2)
        self.xfolds = xfolds
        self.evaluation = evaluation

        # Build an empty pandas dataframe to store the meta results to.
        # As many rows as the folds data, as many cols as base classifiers
        self.colnames = ["v" + str(n) for n in range(len(self.base_classifiers))]
        # Check we have as many colnames as classifiers
        self.stacking_train = pd.DataFrame(np.nan, index=self.xfolds.index, columns=self.colnames)

    def fit(self, X, y, **kwargs):
        """ A generic fit method for meta stacking.

        :param X: A train dataset
        :param y: A train labels
        :param kwargs: Any optional params to give the fit method, i.e in xgboost we may use eval_metirc='auc'
        :return:
        """
        # Loop over the different classifiers.
        n_folds = self.xfolds.ix[:,1].unique()

        for model_no in range(len(self.base_classifiers)):
            print("Running Model ", model_no+1, "of", len(self.base_classifiers))
            for j in range(1, len(n_folds)+1):
                idx0 = self.xfolds[self.xfolds.ix[:,1] != j].index
                idx1 = self.xfolds[self.xfolds.fold5 == j].index
                x0 = X[X.index.isin(idx0)]
                x1 = X[X.index.isin(idx1)]
                y0 = y[y.index.isin(idx0)]
                y1 = y[y.index.isin(idx1)]
                self.base_classifiers[model_no].fit(x0, y0, **kwargs)
                predicted_y_proba = self.base_classifiers[model_no].predict_proba(x1)[:, 1]
                if self.evaluation is not None:
                    print("Current Loss = ", self.evaluation(y1, predicted_y_proba))
                self.stacking_train.ix[self.stacking_train.index.isin(idx1), model_no] = predicted_y_proba
            # Finally fit against all the data
            self.base_classifiers[model_no].fit(X, y, **kwargs)

    def predict_proba(self, X):
        """
        :param X: The data to apply the fitted model from fit
        :return: The predicted_proba of the different classifiers
        """
        stacking_predict_data = pd.DataFrame(np.nan, index=X.index, columns=self.colnames)

        for model_no in range(len(self.base_classifiers)):
            stacking_predict_data.ix[:, model_no] = self.base_classifiers[model_no].predict_proba(X)[:, 1]
        return stacking_predict_data

    @property
    def meta_train(self):
        """
        A return method for the underlying meta data prediction from the training data set as a pandas dataframe
        Use the predict_proba method to score new data for each classifier.
        :return: A pandas dataframe object of stacked predictions of predict_proba[:, 1]
        """
        return self.stacking_train.copy()