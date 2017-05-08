# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:22:54 2017
@author: msiwek
Kaggle - Leaf competition

basic approach using sklearn and KNN
testing different validation strategies

tools: sklearn
model: KNN
hyper: GridSearchCV
validation: (1) inner and outer CV (2) hold-out validation set
cross-validation:
    (1_1) StratifiedShuffleSplit
    (1_2) StratifiedKFold (shuffle=True) + generating oof predictions in the outer CV
    (2_1) StratifiedShuffleSplit, .1 hold-out set
    (2_2) ditto, .2 hold_out set
"""
print(__doc__)
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# data collection

# import sys; sys.path[0] # works when running the script
# abspath = os.path.abspath(__file__); dname = os.path.dirname(abspath); os.chdir(dname)
os.getcwd()
os.chdir('/home/michal/Dropbox/cooperation/_python/Leaf-competition/models')
os.chdir('D:\\data\\Dropbox\\cooperation\\_python\\Leaf-competition\\Models')
os.chdir('G:\\Dropbox\\cooperation\\_python\\Leaf-competition\\Models')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# exploration 1

train.shape # 990, 194
# train.species
train.dtypes
train.species.unique()
train.isnull().sum().sum()
train.describe()
train.columns.groupby(train.dtypes)
test.shape # 594, 193

# preprocessing 1
train_id = train.pop('id')
train_y = train.pop('species')
test_id = test.pop('id')

# 10 observations per class
train_y.unique().shape[0] # 99 classes, and only 990 observations
train_y.value_counts().min()
train_y.value_counts().max()
# -> so exactly 10 observations per class

# no point in trying to plot that fact
sns.countplot(train_y)
#sns.countplot(train_y, color='b')
#plt.plot(train_y.value_counts())
#plt.hist(train_y)

# preprocessing 2
le = LabelEncoder()
y = le.fit_transform(train_y)
labels = le.classes_




''' (1_1) validation schema 1 (outer cross-validation) '''

# cross-validation schema
StratifiedShuffleSplit()
# -> by default
# 10 folds
# 0.1 test size -> each class is represented only by 1 observation in the test set, so there are only 10 ways to split the data into train and test to preserve the class percentages
# but remember: ShuffleSplit samples the train/test set independently in each iteration
scipy.misc.comb(10, 2)
scipy.misc.comb(10, 8)
# -> 45 possibilities if the test set is 0.2, let's do it
sss1 = StratifiedShuffleSplit(test_size=.2)

# tuning and fitting the models

knn = KNeighborsClassifier()
params = {'n_neighbors':[2,3,5,7], 'n_jobs':[-1]}
gs = GridSearchCV(knn, params, cv=sss1, n_jobs=-1)
gs.fit(train, y) # best k=3, (0.887 for test_set=.1) (0.881 for test_set=.2)
print("The best parameters are %s with a score of %0.3f" % (gs.best_params_, gs.best_score_))

# validation (outer cross-validation)

# using default settings
scores = cross_val_score(gs, train, y) # why this doesn't work?
# -> be default StratifiedKFold, n_splits=3, preserving % of classes
# -> it worked!
scores
# scores = np.array([ 0.86616162,  0.85858586,  0.87878788]) # mean 0.868
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), 2*scores.std(ddof=1)))
# -> (0.868 for test_set=.2)
#scores = cross_val_score(gs.estimator, train, y) # this uses knn with default param. values, also requiers fitting it first
# using the same CV method
scores2 = cross_val_score(gs, train, y, cv=sss1, n_jobs=-1)
scores2
print("Accuracy: %0.3f (+/- %0.3f)" % (scores2.mean(), 2*scores2.std(ddof=1)))
# -> (0.871 for test_set=.2)

# prediction

# labels
pred = gs.predict(test)
pred_labels = le.inverse_transform(pred)
# probabilities
pred_prob = gs.predict_proba(test)
# submission
submission = pd.DataFrame(pred_prob, columns=labels)
# np.unique(submission[0:1].values) # just checking
submission.insert(0, 'id', test_id)
# submission.reset_index() # not needed here
submission.to_csv('sLeaf1.csv', index = False)
# -> Your submission scored 1.28887

''' (1_2) validation schema 1 (nested CV but this time preserving the outer predictions) '''

# note: while doing the outer CV you can't use ShuffleSplit as it samples observations into train and test set in each iteration (so you may have repetitions between iterations, or omit some observations in all iterations)
# you have to use KFold with shuffle=True, as then you have the shuffling of all data first and then standard KFold done (so that there are no repetitions and all observations are used)

skf1_2_inner = StratifiedKFold(n_splits=5, shuffle=True) # 8 obs. per class, select 2 for testing
knn = KNeighborsClassifier()
params = {'n_neighbors':[2,3,5,7], 'n_jobs':[-1]}
gs = GridSearchCV(knn, params, cv=skf1_2_inner, n_jobs=-1)
gs.fit(train, y) # best k=3, (0.887 for test_set=.1) (0.881,0.864 for test_set=.2)
print("The best parameters are %s with a score of %0.3f" % (gs.best_params_, gs.best_score_))
# validation
skf1_2_outer = StratifiedKFold(n_splits=5, shuffle=True) # 8 obs. per class, select 2 for testing
pred = cross_val_predict(gs.best_estimator_, train, y, cv=skf1_2_outer, n_jobs=-1)
# -> works ok!
accuracy_score(y, pred) # 0.8687









''' (2_1) validation schema 2 (hold-out validation set) '''

# this is difficult as when you hold out 10% of data you have each class represented by just 1 observation in the validation set; then you just do a prediction for this one class

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.1, stratify=y) # stratify to have all classes represented
y_test.shape[0] # 99 cases
np.unique(y_test).shape[0] # 99 so every observation is from different class and each class is represented by one observation only

# cross-validation schema
# you have 9 observation per class so say 2/9 should be the test set size
sss2_1 = StratifiedShuffleSplit(test_size=2/9)

# tuning and fitting the models
knn = KNeighborsClassifier()
params = {'n_neighbors':[2,3,5,7], 'n_jobs':[-1]}
gs = GridSearchCV(knn, params, cv=sss2_1, n_jobs=-1)
gs.fit(X_train, y_train) # best k=3, 0.887
print("The best parameters are %s with a score of %0.3f" % (gs.best_params_, gs.best_score_))
# -> The best parameters are {'n_neighbors': 3, 'n_jobs': -1} with a score of 0.859

# validation
pred = gs.predict(X_test)
accuracy_score(y_test, pred)
# -> 0.91919 so very high, even higher than CV grid search
# you should increase the size of the validation set... but then you will have less data to train and fit the model in the first place
# conclusion: for this data one should use the nested cross-validation to evaluate the model performance

''' (2_2) validation 2 (but using a bigger hold-out set) '''

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, stratify=y)
y_test.shape[0] # 2*99
np.unique(y_test).shape[0] # 99
tmp = np.bincount(y_test)
tmp.min()
tmp.max() # so 2 obs. per each class
sss2_2 = StratifiedShuffleSplit(test_size=2/8) # 8 obs. per class, select 2 for testing
knn = KNeighborsClassifier()
params = {'n_neighbors':[2,3,5,7], 'n_jobs':[-1]}
gs = GridSearchCV(knn, params, cv=sss2_2, n_jobs=-1)
gs.fit(X_train, y_train) # best k=3, 0.853
print("The best parameters are %s with a score of %0.3f" % (gs.best_params_, gs.best_score_))
# validation
pred = gs.predict(X_test)
accuracy_score(y_test, pred)
# -> 0.863636 so still higher than the CV result but more rational
# classification report
classification_report(y_test, pred)















############## TRASH

train_predictions = gs.predict(train)
acc = accuracy_score(labels, train_predictions)
print("Accuracy: {:.4%}".format(acc)) # 0.95 (in-sample)
test_predictions = gs.predict_proba(test)
submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()
submission.to_csv('sDelaneyExt.csv', index = False)
submission.tail()
# score of this submission: 1.37956






