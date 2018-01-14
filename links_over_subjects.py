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

sns.set_context('poster')
ts_fake = np.zeros([30, 10, 116, 130])
mat = loadmat('/home/andrea/Work/vicente/EC_corrFC_30subj10sess.mat')
fc = np.ravel(mat['corrFC'])
ec = np.ravel(mat['EC'])
mask_AAL = np.array(loadmat('/home/andrea/Work/vicente/mask_EC_AAL.mat')['mask_EC'], dtype=bool)
datasetB = test_retest_dataset(ts_fake, conditions=None, SC=mask_AAL)
#movie.estimate_FC()
datasetB.estimate_EC(subjects=range(30), sessions=range(10), saved='/home/andrea/Work/vicente/EC_corrFC_30subj10sess.mat')
# TODO: estimate EC and compare with saved estimates
datasetB.subject_classif = classification('subejcts', 'EC',
                                       datasetB.make_target_subjects(),
                                       datasetB.make_data_matrix(C='EC'))
subj_n = [2, 5, 10, 15, 20, 25, 30]
# classification varying subjects already done and saved in prova2.pickle
datasetB.subject_classif.classify_over_subjects(n_subjects=subj_n, repetitions=10, extract_features=False)
#movie.subject_classif.classify_over_sessions(n_sessions=[30, 60])
#pickle.dump(datasetB, open("/home/andrea/Work/code/neuroconn/prova2.pickle", "wb"))
datasetB = pickle.load(open("/home/andrea/Work/code/neuroconn/prova2.pickle", "rb"))

# plot number of minimal features over number of subjects
n_feat_subj = np.zeros([len(subj_n), 10])
for s, nsub in enumerate(subj_n):
    for r in range(10):
        n_feat_subj[s, r] = datasetB.subject_classif.features_extraction[nsub][r]['score'].shape[0] - 1
sns.set_palette('colorblind')
plt.figure()
plt.fill_between(subj_n, n_feat_subj.mean(axis=1) + n_feat_subj.std(axis=1),
                 n_feat_subj.mean(axis=1) - n_feat_subj.std(axis=1), alpha=0.5, color='grey')
plt.scatter(subj_n, n_feat_subj.mean(axis=1), label="data (mean)", color='grey')
plt.xlabel('# subjects')
plt.ylabel('minimal features')

# fit minimal features over subjects with different functions
xx = np.repeat(subj_n, 10)
yy = n_feat_subj.flatten()
from scipy.optimize import curve_fit
def lin_f(x, a, b):
    return x * b + a
def log_f(x, a, b):
    return np.log(x) * b + a
def pow_f(x, a, b):
    return x**b * a
popt_lin, pcov_lin = curve_fit(lin_f, xx, yy)
popt_log, pcov_log = curve_fit(log_f, xx, yy)
popt_pow, pcov_pow = curve_fit(pow_f, xx, yy)

# calculate Sum of Squared Errors
SSE_lin = 0
SSE_log = 0
SSE_pow = 0
for i, x in enumerate(xx):
    SSE_lin += (lin_f(x, popt_lin[0], popt_lin[1]) - yy[i])**2
    SSE_log += (log_f(x, popt_log[0], popt_log[1]) - yy[i])**2
    SSE_pow += (pow_f(x, popt_pow[0], popt_pow[1]) - yy[i])**2

# plot fitted functions
xx2 = np.linspace(xx.min(), xx.max(), 100)
plt.plot(xx2, lin_f(xx2, popt_lin[0], popt_lin[1]), label=r"$a + b x$, SSE: %d" % (SSE_lin))
plt.plot(xx2, log_f(xx2, popt_log[0], popt_log[1]), label=r"$a+ln(x) b$, SSE: %d" % (SSE_log))
plt.plot(xx2, pow_f(xx2, popt_pow[0], popt_pow[1]), label=r"$a x^b$, SSE: %d" % (SSE_pow))
plt.legend()

#%%
# fit classification accuracy over subjects with different functions
xx = np.repeat(subj_n, 10)
yy = datasetB.subject_classif.score_over_subjects.flatten()
from scipy.optimize import curve_fit
def lin_f(x, a, b):
    return x * b + a
def log_f(x, a, b):
    return np.log(x) * b + a
def pow_f(x, a, b):
    return x**b * a
popt_lin, pcov_lin = curve_fit(lin_f, xx, yy)
popt_log, pcov_log = curve_fit(log_f, xx, yy)
popt_pow, pcov_pow = curve_fit(pow_f, xx, yy)

# calculate Sum of Squared Errors
SSE_lin = 0
SSE_log = 0
SSE_pow = 0
for i, x in enumerate(xx):
    SSE_lin += (lin_f(x, popt_lin[0], popt_lin[1]) - yy[i])**2
    SSE_log += (log_f(x, popt_log[0], popt_log[1]) - yy[i])**2
    SSE_pow += (pow_f(x, popt_pow[0], popt_pow[1]) - yy[i])**2

# plot fitted functions
plt.figure()
plt.scatter(subj_n, datasetB.subject_classif.score_over_subjects.mean(axis=1),
            color='grey', label="data (mean)")
plt.fill_between(subj_n,
                         datasetB.subject_classif.score_over_subjects.mean(axis=1) +
                         datasetB.subject_classif.score_over_subjects.std(axis=1),
                         datasetB.subject_classif.score_over_subjects.mean(axis=1) -
                         datasetB.subject_classif.score_over_subjects.std(axis=1),
                         alpha=0.5, color='grey')
xx2 = np.linspace(xx.min(), xx.max(), 100)
plt.plot(xx2, lin_f(xx2, popt_lin[0], popt_lin[1]), '--', label=r"$a + b x$, SSE: %.3f" % (SSE_lin), color='black')
plt.plot(np.linspace(2, 1000, 100), log_f(np.linspace(2, 1000, 100), popt_log[0], popt_log[1]), label=r"$a+ln(x) b$, SSE: %.3f" % (SSE_log))
#plt.plot(xx2, pow_f(xx2, popt_pow[0], popt_pow[1]), label=r"$a x^b$, SSE: %.3f" % (SSE_pow))
plt.legend()
plt.xlabel('# subjects')
plt.ylabel('test-set accuracy')

#%% plot correlation of rankings for each repetition for a given number of subjects
subjN = 5
rr = np.zeros([10, len(datasetB.subject_classif.features_extraction[subjN][0]['ranking'])])
for r in range(10):
    rr[r,:] = datasetB.subject_classif.features_extraction[subjN][r]['ranking']
rm = np.corrcoef(rr)
rm[np.eye(10, dtype=bool)] = 0
plt.figure()
sns.heatmap(rm)


