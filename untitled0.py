#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:30:06 2017

@author: andrea
"""

import numpy as np
from MOU_estimation import MOU_Lyapunov
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


class subject:
    """
    Basic class for subject
    """
    def __init__(self, ID, SC=None):
        self.sessions = list()
        self.SC = SC
        self.ID = ID


class session:
    """
    Basic class for session
    """
    def __init__(self, data, condition, time):
        self.BOLD = data
        self.condition_ID = condition
        self.time = time


class classification:
    """
    Basic class to collect everything related to a classification
    e.g. scores over sessions, subjects, etc., confusion matrix, features,
    etc., etc.
    """
    def __init__(self, name):
        self.name = "Something to tell what you are classifying"


class test_retest_dataset:
    """
    Basic class for test-retest dataset.
    """

    def __init__(self, BOLD_ts, conditions=None, SC=None, time=None):
        self.n_subjects = np.shape(BOLD_ts)[0]
        self.n_sessions = np.shape(BOLD_ts)[1]
        self.n_ROIs = np.shape(BOLD_ts)[2]
        self.n_time_samples = np.shape(BOLD_ts)[3]
        self.subjects = list()
        for sb in range(self.n_subjects):
            self.subjects.append(subject(sb))
            for ss in range(self.n_sessions):
                if conditions is None:  # assign all 1s if no condition
                    cond = 1
                elif isinstance(conditions, (list, np.ndarray)):
                    cond = conditions[ss]
                elif type(conditions) is dict:
                    # TODO: implement this!
                    raise ValueError("Sorry dictionary is not yet supported.")
                else:
                    raise ValueError("conditions has to be a list",
                                     "or numpy array.")
                # TODO: add same tests for time
                self.subjects[sb].sessions.append(session(BOLD_ts[sb, ss, :, :], condition=cond, time=time))

    def estimate_EC(self, subjects, sessions, norm_fc=None):
        # TODO: modify to produce fitting graphics in a separate folder
        #       if required
        for sb in subjects:
            for ss in sessions:
                BOLD_ts = self.subjects[sb].sessions[ss].BOLD
                SC = self.subjects[sb].SC
                EC, S, tau_x, d_fit = MOU_Lyapunov(ts_emp=BOLD_ts,
                                                   SC_mask=SC, norm_fc=norm_fc)
                self.subjects[sb].sessions[ss].EC = EC
                self.subjects[sb].sessions[ss].Sigma = S
                self.subjects[sb].sessions[ss].tau_x = tau_x
                self.subjects[sb].sessions[ss].model_fit = d_fit

    def estimate_FC(self, subjects=None, sessions=None):
        if subjects is None:
            subjects = range(self.n_subjects)
        if sessions is None:
            sessions = range(self.n_sessions)
        for sb in subjects:
            for ss in sessions:
                BOLD_ts = self.subjects[sb].sessions[ss].BOLD
                FC = np.corrcoef(BOLD_ts)
                self.subjects[sb].sessions[ss].FC = FC

    def classify_over_sessions(self):
        return 0

    def classify_over_subjects(self):
        return 0

    def classify_over_time(self):
        return 0

    def confusion_matrix(self):
        return 0

    def extract_features(self):
        return 0

    def minimal_VS_random(self):
        return 0

    def minimalBest_VS_minimalWorst(self):
        return 0

    def make_data_matrix(self, subjects=None, sessions=None, C='FC'):
        # move to utils
        """
        Builds a data matrix [n_samples, n_features] by concatenating
        elements of connectivity matrix C and stacking them over rows
        for each subject and session specified in subjects and sessions.
        """
        if subjects is None:
            subjects = range(self.n_subjects)
        if sessions is None:
            sessions = range(self.n_sessions)
        if C is 'FC':
            idxs = np.triu_indices(self.n_ROIs)
            X = np.zeros([len(subjects)*len(sessions), len(idxs[0])])
            i = 0
            for sb in subjects:
                for ss in sessions:
                    X[i, :] = self.subjects[sb].sessions[ss].FC[idxs]
                    i += 1
        # TODO implement test for EC
        else:
            raise ValueError("C has to be either FC or EC.")
        return X

    def make_target_subjects(self, subjects=None, sessions=None):
        # move to utils
        """
        Builds a target vector y with subjects ID to classify subjects.
        """
        if subjects is None:
            subjects = range(self.n_subjects)
        if sessions is None:
            sessions = range(self.n_sessions)
        y = np.zeros([len(subjects)*len(sessions)])
        i = 0
        for sb in subjects:
                for ss in sessions:
                    y[i] = self.subjects[sb].ID
                    i += 1
        return y


def crossvalidate_clf(X, y, train_size, repetitions=10):
    # TODO: move to utils
    clf = LogisticRegression(C=10000, penalty='l2',
                             multi_class='multinomial',
                             solver='lbfgs')
    pipe = Pipeline([('pca', PCA()),
                     ('clf', clf)])
    scores = np.zeros([repetitions])
    sss = StratifiedShuffleSplit(n_splits=repetitions, test_size=None,
                                 train_size=train_size, random_state=0)
    r = 0  # repetitions index
    for train_idx, test_idx in sss.split(X, y):
        data_train = X[train_idx, :]
        y_train = y[train_idx]
        data_test = X[test_idx, :]
        y_test = y[test_idx]
        # fit clf on train
        pipe.fit(data_train, y_train)
        # predict on test
        scores[r] = pipe.score(data_test, y_test)
        r += 1
    return scores


def calc_mean_FC():
    # TODO: move to utils
    return 0


ts_emp = np.load('/home/andrea/Work/matt_movie_scripts/EC_estimation/rest_movie_ts.npy')
movie = test_retest_dataset(ts_emp, conditions=[0, 0, 1, 1])
movie.estimate_FC()
X = movie.make_data_matrix()
y = movie.make_target_subjects()
s = crossvalidate_clf(X, y, train_size=22)
