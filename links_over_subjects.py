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
movie.subject_classif.classify_over_subjects(n_subjects=[2, 4], repetitions=10, extract_features=True)
#movie.subject_classif.classify_over_sessions(n_sessions=[30, 60])
