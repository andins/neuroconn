import numpy as np
from MOU_estimation import MOU_Lyapunov
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from untitled0 import test_retest_dataset, classification
from scipy.io import loadmat
import pickle

# TODOs: many many stuff

ts_fake = np.zeros([30, 10, 116, 130])
mat = loadmat('/home/andrea/Work/vicente/EC_corrFC_30subj10sess.mat')
fc = np.ravel(mat['corrFC'])
ec = np.ravel(mat['EC'])
mask_AAL = np.array(loadmat('/home/andrea/Work/vicente/mask_EC_AAL.mat')['mask_EC'], dtype=bool)
movie = test_retest_dataset(ts_fake, conditions=None, SC=mask_AAL)
#movie.estimate_FC()
movie.estimate_EC(subjects=range(30), sessions=range(10), saved='/home/andrea/Work/vicente/EC_corrFC_30subj10sess.mat')
movie.subject_classif = classification('subejcts', 'EC',
                                       movie.make_target_subjects(),
                                       movie.make_data_matrix(C='EC'))
subj_n = [2, 5, 10, 15, 20, 25, 30]
movie.subject_classif.classify_over_subjects(n_subjects=subj_n, repetitions=10, extract_features=True)
#movie.subject_classif.classify_over_sessions(n_sessions=[30, 60])
pickle.dump(movie, open("/home/andrea/Work/code/neuroconn/prova2.pickle", "wb"))
n_feat_subj = np.zeros([len(subj_n), 10])
for s, nsub in enumerate(subj_n):
    for r in range(10):
        n_feat_subj[s, r] = movie.subject_classif.features_extraction[nsub][r].shape[0] - 1
plt.figure()
plt.fill_between(subj_n, n_feat_subj.mean(axis=1) + n_feat_subj.std(axis=1),
                 n_feat_subj.mean(axis=1) - n_feat_subj.std(axis=1), alpha=0.5)
plt.plot(subj_n, n_feat_subj.mean(axis=1))
plt.xlabel('# subjects')
plt.ylabel('minimal features')

