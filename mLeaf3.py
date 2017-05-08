# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:30:37 2017
@author: a
Kaggle - Leaf competition

Topic: ensembling basic classifiers by VotingClassifier (hard)

Models: KNN, DecisionTree, SVC
Ensambling: VotingClassifier
Tuning: GridSearchCV
CV: inner/outer: StratifiedKFold

Conclusions
- use better individual classifiers
- check for correlation between classifiers
- set weights depending on individual performance
- try better ensembling methods
"""

print(__doc__)

import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Loading data
os.getcwd()
os.chdir('/home/michal/Dropbox/cooperation/_python/Leaf-competition/models')
os.chdir('D:\\data\\Dropbox\\cooperation\\_python\\Leaf-competition\\Models')
os.chdir('G:\\Dropbox\\cooperation\\_python\\Leaf-competition\\Models')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Preprocessing
train_id = train.pop('id')
train_y = train.pop('species')
test_id = test.pop('id')
le = LabelEncoder()
y = le.fit_transform(train_y)
n_samples, n_features = train.shape

# Inner CV
skf = StratifiedKFold(5, shuffle=True) # 10 obs. per class, select 2 for testing

# Training classifiers
clf1 = DecisionTreeClassifier() #max_depth=4)
clf2 = KNeighborsClassifier() #n_neighbors=7)
clf3 = SVC(kernel='rbf') #, probability=True)
estimators=[('dt', clf1), ('knn', clf2),('svc', clf3)]
n_estimators = len(estimators)
eclf = VotingClassifier(estimators)#, n_jobs=-1) #voting='soft', weights=[2, 1, 2])
pipeline = Pipeline([('vc', eclf)])

""" 1_1 simple fitting without GridSearch tuning """
# fitting (ensemble)
pipeline.fit(train, y)
# validation (ensemble)
scores = cross_val_score(pipeline, train, y, cv=skf, n_jobs=-1)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), 2*scores.std(ddof=1)))
"""
Accuracy: 0.849 (+/- 0.027)
Accuracy: 0.859 (+/- 0.036)
Accuracy: 0.862 (+/- 0.034)
"""
# CV score of all classifiers and the ensemble (before GridSearch tuning)
for clf, label in zip([clf1, clf2, clf3, eclf], ['Decision Tree', 'KNN', 'SVC-RBF', 'Ensemble']):
    scores = cross_val_score(clf, train, y, cv=skf, scoring='accuracy')
    print("Accuracy: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), 2*scores.std(ddof=1), label))
"""
Accuracy: 0.663 (+/- 0.019) [Decision Tree]
Accuracy: 0.858 (+/- 0.031) [KNN]
Accuracy: 0.794 (+/- 0.031) [SVC-RBF]
Accuracy: 0.862 (+/- 0.054) [Ensemble]
"""

""" 1_2 tuning hyperparameters with GridSearch """
g = 1/n_features
parameters = {'vc__dt__max_depth':[5,10,None],
              'vc__knn__n_neighbors':[2,3,5],
              'vc__knn__n_jobs':[-1],
              'vc__svc__gamma':[10*g, g, g/10]}
gs = GridSearchCV(pipeline, parameters, cv=skf, n_jobs=-1)
gs.fit(train, y)
print("The best parameters are %s with a score of %0.3f" % (gs.best_params_, gs.best_score_))
"""
The best parameters are
{'vc__dt__max_depth': None,
'vc__svc__gamma': 0.0005208333333333333,
'vc__knn__n_jobs': -1,
'vc__knn__n_neighbors': 3} (sometimes it's 2)
with a score of 0.882
"""
# Explore the individual estimators
gs.best_estimator_
gs.best_estimator_.named_steps['vc']
gs.best_estimator_.named_steps['vc'].estimators_
gs.best_estimator_.named_steps['vc'].estimators_[0]
# CV score for individual classifiers (after GridSearch tuning)
for i in np.arange(n_estimators):
    scores = cross_val_score(gs.best_estimator_.named_steps['vc'].estimators_[i],
                             train, y, cv=skf, scoring='accuracy')
    print("Accuracy: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), 2*scores.std(ddof=1),
                            estimators[i][1].__class__.__name__))
"""
Accuracy: 0.65 (+/- 0.04) [DecisionTreeClassifier]
Accuracy: 0.88 (+/- 0.06) [KNeighborsClassifier]
Accuracy: 0.79 (+/- 0.06) [SVC]

Accuracy: 0.673 (+/- 0.074) [DecisionTreeClassifier]
Accuracy: 0.871 (+/- 0.023) [KNeighborsClassifier]
Accuracy: 0.797 (+/- 0.042) [SVC]

So actually only KNN improved through grid search, not much gain for other classifiers.
"""
# validation (ensemble after GridSearch tuning) (remember to use the .best_estimator_)
scores = cross_val_score(gs.best_estimator_, train, y, cv=skf, n_jobs=-1)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), 2*scores.std(ddof=1)))
"""
for gs.best_estimator_: Accuracy: 0.883 (+/- 0.023)
for gs: Accuracy: 0.876 (+/- 0.061)
"""
