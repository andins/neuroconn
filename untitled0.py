#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:30:06 2017

@author: andrea
"""

import numpy as np
from MOU_estimation import MOU_Lyapunov


class subject:
    """
    Basic class for subject
    """
    def __init__(self, SC=None):
        self.sessions = list()
        self.SC = SC


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
            self.subjects.append(subject())
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
                self.subjects[sb].sessions.append(session(BOLD_ts[sb, ss, :, :], condition=cond), time=time)

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


def split_train_test_same(X, y, train_size=0.8):
    """
    Split data in train and test keeping the approximate proportion of
    samples for each class
    """
    # TODO: move to utils
    train_idx = 0
    test_idx = 0
    return train_idx, test_idx


def crossvalidate_clf(X, y):
    # TODO: move to utils
    # for repetitions
        # split_train_test_same()
        # fit clf on train
        # predict on test
    return 0


def calc_mean_FC():
    # TODO: move to utils
    return 0


ts_emp = np.load('/home/andrea/Work/matt_movie_scripts/EC_estimation/rest_movie_ts.npy')
movie = test_retest_dataset(ts_emp, conditions=[0, 0, 1, 1])
