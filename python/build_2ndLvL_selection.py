import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bortua2 import BorutaPy2

# We need to import all the meta based predictions from ./metafeatures and
# combine these into a single pandas dataframe.
# Once this has been done we are then able to test different feature selection
# methods, i.e bortua or linear combination reductions for the second level
# metas.

# load X and y
X = pd.read_csv('my_X_table.csv', index_col=0).values
y = pd.read_csv('my_y_vector.csv', index_col=0).values


# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy2(rf, n_estimators='auto', verbose=2)

# find all relevant features
feat_selector.fit(X, y)

# check selected features
feat_selector.support_

# check ranking of features
feat_selector.ranking_
