# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 20:24:54 2017
@author: a
Kaggle - Leaf competition

concise extract from mLeaf1 + using a pipeline

knn + GridSearchCV + inner/outer CV
"""
print(__doc__)

import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# data collection
os.getcwd()
os.chdir('/home/michal/Dropbox/cooperation/_python/Leaf-competition/models')
os.chdir('D:\\data\\Dropbox\\cooperation\\_python\\Leaf-competition\\Models')
os.chdir('G:\\Dropbox\\cooperation\\_python\\Leaf-competition\\Models')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# initial preprocessing
train_id = train.pop('id')
train_y = train.pop('species')
test_id = test.pop('id')
le = LabelEncoder()
y = le.fit_transform(train_y)
# CV
skf = StratifiedKFold(5, shuffle=True) # 10 obs. per class, select 2 for testing


''' running and tuning knn model '''

# KNN
knn = KNeighborsClassifier()
params = {'n_neighbors':[2,3,5,7], 'n_jobs':[-1]}
gs = GridSearchCV(knn, params, cv=skf, n_jobs=-1)
gs.fit(train, y) # best k=3, (0.874)
print("The best parameters are %s with a score of %0.3f" % (gs.best_params_, gs.best_score_))
# validation
scores = cross_val_score(gs, train, y, cv=skf, n_jobs=-1)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), 2*scores.std(ddof=1))) # (0.880)


''' the same as a pipeline '''

from sklearn.pipeline import Pipeline
pipeline = Pipeline([('knn', KNeighborsClassifier())])
parameters = {'knn__n_neighbors':[2,3,5,7],
              'knn__n_jobs':[-1]}
gs = GridSearchCV(pipeline, parameters, cv=skf, n_jobs=-1)
gs.fit(train, y) # best k=3
print("The best parameters are %s with a score of %0.3f" % (gs.best_params_, gs.best_score_))
# validation
scores = cross_val_score(gs, train, y, cv=skf, n_jobs=-1)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), 2*scores.std(ddof=1)))
