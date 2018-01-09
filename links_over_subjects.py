import numpy as np
from MOU_estimation import MOU_Lyapunov
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from untitled0 import test_retest_dataset, classification

# TODOs: many many stuff

ts_emp = np.load('/home/andrea/Work/matt_movie_scripts/EC_estimation/rest_movie_ts.npy')
movie = test_retest_dataset(ts_emp, conditions=None)
#movie.estimate_FC()
movie.estimate_EC()
movie.subject_classif = classification('subejcts', 'FC',
                                       movie.make_target_subjects(),
                                       movie.make_data_matrix())
movie.subject_classif.classify_over_subjects(extract_features=True)
