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
    def __init__(self, data, condition):
        self.BOLD = data
        self.condition_ID = condition


class test_retest_dataset:
    """
    Basic class for test-retest dataset.
    """

    def __init__(self, BOLD_ts, conditions=None, SC=None):
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
                self.subjects[sb].sessions.append(session(BOLD_ts[sb, ss, :, :], condition=cond))

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
            
    def estimate_FC(self, subjects, sessions):
        return 0


def calc_mean_FC(self):
    # TODO: move to utils
    return 0


ts_emp = np.load('/home/andrea/Work/matt_movie_scripts/EC_estimation/rest_movie_ts.npy')
movie = test_retest_dataset(ts_emp, conditions=[0, 0, 1, 1])
