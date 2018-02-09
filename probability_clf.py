#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:59:16 2018

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from untitled0 import test_retest_dataset, classification, crossvalidate_clf

sns.set_context('poster')


##############################
# DATASET C (movie)
ts_movie = np.load('/home/andrea/Work/vicente/data/movie/ts_emp.npy')    
# remove bad subjects: 1 11 19
ts_clean = np.delete(ts_movie, [0, 10, 18], 0)

movMask = np.load('/home/andrea/Work/vicente/mask_EC.npy')  # [roi, roi] the mask for existing EC connections
movie = test_retest_dataset(ts_clean, conditions=[0, 0, 1, 1, 1], SC=movMask)
movie.estimate_FC()
# estimate EC
#movie.estimate_EC()
# load EC previously estimated
movie.estimate_EC(subjects=range(19), sessions=range(5), saved='/home/andrea/Work/vicente/EC_movie.npy')
# create a classification object for subject with EC
movie.subject_classif_EC = classification('subejcts', 'EC',
                                       movie.make_target_subjects(),
                                       movie.make_data_matrix())
# create a classification object for condition with EC
movie.condition_classif_EC = classification('conditions', 'EC',
                                         movie.make_target_conditions(),
                                         movie.make_data_matrix())

# ranking will be saved in movie.condition_classif_EC.ranking
#movie.condition_classif_EC.rank_features()
# ranking will be saved in movie.subject_classif_EC.ranking
#movie.subject_classif_EC.rank_features()
# fit MLR and returns the accuracy and the classifier object
X = movie.subject_classif_EC.X[:, :]  # index second axis to use a subset of features
# get the test-set accuracy and classifier object
score_s, clf_s = crossvalidate_clf(X, movie.subject_classif_EC.y,
                                   train_size=76, repetitions=10)
clf_s.fit(X, movie.subject_classif_EC.y)  # refit the classifier with whole dataset
# access classifier prediction of class probability
prob_s = clf_s.predict_proba(movie.subject_classif_EC.X)
# fit MLR and returns the accuracy and the classifier object
X = movie.condition_classif_EC.X[:, :]  # index second axis to use a subset of features
# get the test-set accuracy and classifier object
score_c, clf_c = crossvalidate_clf(X, movie.condition_classif_EC.y,
                                   train_size=.8, repetitions=10)
clf_c.fit(X, movie.condition_classif_EC.y)  # refit the classifier with whole dataset
# access classifier prediction of class probability
prob_c = clf_c.predict_proba(movie.condition_classif_EC.X)

plt.figure()
plt.plot(prob_s[0:20:5].T)
