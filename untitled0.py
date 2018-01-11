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
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import pickle


class subject:
    """
    Basic class for subject
    """
    def __init__(self, ID, SC=None):
        self.sessions = list()
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
    def __init__(self, target_name, features_name, y=None, X=None):
        """
        If X and y are not provided they are initialized to None and their
        value should be changed before actually running any classification.
        """
        self.target_name = target_name
        self.features_name = features_name
        self.y = y
        self.X = X

    def classify_over_sessions(self, n_sessions, repetitions=10):
        """
        """
        self.score_over_sessions = np.zeros([len(n_sessions), repetitions])
        for s, ses in enumerate(n_sessions):
            self.score_over_sessions[s, :] = crossvalidate_clf(self.X, self.y,
                                                 train_size=ses,
                                                 repetitions=repetitions)
        # TODO: probably write a function to make the plot
        plt.figure()
        plt.fill_between(n_sessions,
                         self.score_over_sessions.mean(axis=1) +
                                                      self.score_over_sessions.std(axis=1),
                         self.score_over_sessions.mean(axis=1) -
                                                      self.score_over_sessions.std(axis=1),
                         alpha=0.5)
        plt.plot(n_sessions, self.score_over_sessions.mean(axis=1))
        plt.xlabel('# sessions')
        plt.ylabel('CV score')

    def classify_over_subjects(self, n_subjects, repetitions=10):
        """
        Performs classification with a subset of subjects, e.g. to investigate the relationship between
        number of subjects and classification performance.
        Nothing is returned but a property is created in the classification object to store the classification score.
        A figure is produced that represents the classification score as a function of the number of subjects.
        PARAMETERS:
            n_subjects is a list-like object with the number of subject to include in the classification.
            repetitions is an integer that controls the number of times that the classification is repeated
            for each element in n_subjects. New random subjects and sessions are drawn at each repetition.
        """
        # TODO: add the possibility to extract features for each subset of subjects
        
        self.score_over_subjects = np.zeros([len(n_subjects), repetitions])
        for s, sub in enumerate(n_subjects):
            subj_labels = np.unique(self.y)  # get labels of the subjects
            for r in range(repetitions):                
                np.random.shuffle(subj_labels)  # shuffle labels
                indxs = subj_labels[0:sub]  # and take first sub labels (this way we get sub random labels)
                idxx = np.zeros(np.shape(self.y), dtype=bool)  # initial 0 index vector
                for i in range(len(indxs)):  # for each subject included in the classification
                    idxx += self.y==indxs[i]  # idxx will be True for each session corresponding to each subject in indxs
                newy = self.y[idxx]  # create a new y for the classification with a subset of subjects
                newX = self.X[idxx, :]  # and a new X
                self.score_over_subjects[s, r] = crossvalidate_clf(newX, newy,
                                                 train_size=sub,
                                                 repetitions=1)
        plt.figure()
        plt.fill_between(n_subjects,
                         self.score_over_subjects.mean(axis=1) +
                                                      self.score_over_subjects.std(axis=1),
                         self.score_over_subjects.mean(axis=1) -
                                                      self.score_over_subjects.std(axis=1),
                         alpha=0.5)
        plt.plot(n_subjects, self.score_over_subjects.mean(axis=1))
        plt.xlabel('# subjects')
        plt.ylabel('CV score')

    def classify_over_time(self):
        return 0

    def compare_classifiers(self):
        return 0

    def confusion_matrix(self):
        return 0

    def rank_features(self, saved=False):
        # Rank each feature in the classification using RFE
        # TODO: save and recalc are just to import rankings calculated outside the object
        # the way to save it is directly saving the TRD object: clean up!
        if saved is False:
            mlr = LogisticRegression(C=10000, penalty='l2', multi_class= 'multinomial', solver='lbfgs')
            rfe = RFE(estimator=mlr, n_features_to_select=1, step=1)
            rfe.fit(self.X, self.y)
            self.ranking = rfe.ranking_
        else:
            rfe = pickle.load(open(saved, "rb"))
            self.ranking = rfe.ranking_ 
    
    def extract_features(self, X, y, repetitions=10, wdw_l=2, tol=0.001, start_with=1, stop_after=50, step=1, fig=True):
        # This function need a ranking to be already calculated with
        # function rank_features.
        # INPUT:
        # X: data
        # y: target
        # window_length of rolling average
        # tolerance for considering the derivative equal to zero
        # start_with: integer determines the starting number of features
        # stop_after: integer to limit the number of features used for a fast inspection (in case many features are needed)
        # step : how many feature to add at each step
        # fig: boolean to plot a figure
        # TODO: add check for ranking
        # TODO: use X and y of the classification object
        
        features2add = np.arange(start_with, stop_after+1, step)
        rfe_scores = np.zeros([len(features2add), repetitions])
        smoothed_scores = np.zeros([len(features2add), repetitions])
        
        for n, nf in enumerate(features2add):  # add features starting from 1 in the order of the ranking and calculate the classification score
            # only calculates performance and updates the number of features if 
            # performance has not yet saturated (deriv < tol in previous step)
            if n<3 or abs(np.gradient(smoothed_scores[0:n])[n-1])>tol:
                subset_feat = self.ranking<=nf
                rfe_scores[n,:] = self.minimal_model_performance(X, y, subset_feat, repetitions=10)
                # smooth scores with a rolling average to get rid of spurious peaks
                smoothed_scores = np.convolve(rfe_scores.mean(axis=1), np.ones((wdw_l,))/wdw_l, mode='same')
                rfe_num_feat = nf
            else:
                break

        if nf==features2add[-1]:
            print("Warning: performance may have not converged: set stop_after to a higher value to explore more features")

        rfe_scores = np.delete(rfe_scores, np.arange(n+1,rfe_scores.shape[0]), axis=0)
        if fig:
            plt.figure()
            plt.errorbar(features2add[0:rfe_num_feat],
                         np.mean(rfe_scores, axis=1),
                         np.std(rfe_scores, axis=1))
            plt.xlabel('# features')
            plt.ylabel('CV score')        
        self.score_over_features = rfe_scores

    def minimal_model_performance(self, X, y, selected_features, repetitions=10):
        # TODO: add a compare parameter with possible values: 'None', 'random', 'worse'
        #       to compare the performance of the minimal model with that of a model with same
        #       number of features chosen at random or from worse ranked.
        X_min = X[:, selected_features]
        score_min = crossvalidate_clf(X_min, y, train_size=0.9,
                                      repetitions=repetitions)
        return score_min


class test_retest_dataset:
    """
    Basic class for test-retest dataset.
    You can estimate EC or FC and calculate
    other stuff useful for classification.
    """

    def __init__(self, BOLD_ts, conditions=None, SC=None, time=None):
        self.n_subjects = np.shape(BOLD_ts)[0]
        self.n_sessions = np.shape(BOLD_ts)[1]
        self.n_ROIs = np.shape(BOLD_ts)[2]
        self.n_time_samples = np.shape(BOLD_ts)[3]
        self.SC = SC
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

    def estimate_EC(self, subjects, sessions, norm_fc=None, saved=False):
        # TODO: modify to produce fitting graphics in a separate folder
        #       if required
        if saved is not False:
            # TODO: change these two lines toallow the more general case of i
            # having the EC directly saved as a numpy array
            mat = loadmat(saved)
            ec = np.ravel(mat['EC'])
        for sb in subjects:
            for ss in sessions:
                if saved is False:
                    BOLD_ts = self.subjects[sb].sessions[ss].BOLD
                    SC = self.SC
                    EC, S, tau_x, d_fit = MOU_Lyapunov(ts_emp=BOLD_ts,
                                                       SC_mask=SC, norm_fc=norm_fc)
                    self.subjects[sb].sessions[ss].Sigma = S
                    self.subjects[sb].sessions[ss].tau_x = tau_x
                    self.subjects[sb].sessions[ss].model_fit = d_fit
                    self.subjects[sb].sessions[ss].EC = EC
                else:
                    EC = np.zeros([self.n_ROIs, self.n_ROIs])
                    EC[self.SC] = ec[sb][ss]
                    self.subjects[sb].sessions[ss].EC = EC

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
        elif C is 'EC':
            num_non_zero = np.sum(self.SC)
            X = np.zeros([len(subjects)*len(sessions), num_non_zero])
            i = 0
            for sb in subjects:
                for ss in sessions:
                    X[i, :] = self.subjects[sb].sessions[ss].EC[self.SC]
                    i += 1
        else:
            raise ValueError("C has to be either FC or EC.")
        return X

    def make_target_subjects(self, subjects=None, sessions=None):
        # TODO: move to utils (?)
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

    def make_target_conditions(self, subjects=None, sessions=None):
        # TODO: move to utils (?)
        """
        Builds a target vector y with conditions ID to classify conditions.
        """
        if subjects is None:
            subjects = range(self.n_subjects)
        if sessions is None:
            sessions = range(self.n_sessions)
        y = np.zeros([len(subjects)*len(sessions)])
        i = 0
        for sb in subjects:
                for ss in sessions:
                    y[i] = self.subjects[sb].sessions[ss].condition_ID
                    i += 1
        return y


def crossvalidate_clf(X, y, train_size, repetitions=10, random_state=None):
    # TODO: move to utils
    # TODO: let choose the classifier
    clf = LogisticRegression(C=10000, penalty='l2',
                             multi_class='multinomial',
                             solver='lbfgs')
    pipe = Pipeline([('clf', clf)])
    scores = np.zeros([repetitions])
    sss = StratifiedShuffleSplit(n_splits=repetitions, test_size=None,
                                 train_size=train_size, random_state=random_state)
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


#ts_emp = np.load('/home/andrea/Work/matt_movie_scripts/EC_estimation/rest_movie_ts.npy')    
#movie = test_retest_dataset(ts_emp, conditions=[0, 0, 1, 1])
#movie.estimate_FC()
##movie.estimate_EC()
#movie.subject_classif = classification('subejcts', 'FC',
#                                       movie.make_target_subjects(),
#                                       movie.make_data_matrix())
#movie.subject_classif.classify_over_sessions(n_sessions=[22, 44, 66])
#movie.condition_classif = classification('conditions', 'FC',
#                                         movie.make_target_conditions(),
#                                         movie.make_data_matrix())
#movie.condition_classif.classify_over_sessions(n_sessions=[4, 10, 22, 66])
