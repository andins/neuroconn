import numpy as np
from MOU_estimation import MOU_Lyapunov
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from untitled0 import test_retest_dataset, classification, crossvalidate_clf
from scipy.io import loadmat
import pickle

sns.set_context('poster')

# DATASET B
# load timeseries
ts_B = np.zeros([30, 10, 116, 295])
for s, sub in enumerate(np.arange(25427, 25457)):
    for e, ses in enumerate(np.arange(1, 11)):
        file_name = 'ROISignals_00{}_SE{:0>2}.mat'.format(sub, ses)
        ts_B[s, e, :, :] = loadmat('/home/andrea/Work/vicente/data/datasetB/' + file_name)['ROISignals'].T
mat = loadmat('/home/andrea/Work/vicente/EC_corrFC_30subj10sess.mat')
fc = np.ravel(mat['corrFC'])
ec = np.ravel(mat['EC'])
mask_AAL = np.array(loadmat('/home/andrea/Work/vicente/mask_EC_AAL.mat')['mask_EC'], dtype=bool)
datasetB = test_retest_dataset(ts_B, conditions=None, SC=mask_AAL)
# estimate FC
datasetB.estimate_FC()
# estimate EC
datasetB.estimate_EC(subjects=range(30), sessions=range(10), saved='/home/andrea/Work/vicente/EC_corrFC_30subj10sess.mat')
# TODO: estimate EC and compare with saved estimates

# classify over subsets of subjects of different size to show
# the trend and the projection to 1000 subjects 
datasetB.subject_classif_EC = classification('subejcts', 'EC',
                                       datasetB.make_target_subjects(),
                                       datasetB.make_data_matrix(C='EC'))
score, clf = crossvalidate_clf(datasetB.subject_classif_EC.X,
                  datasetB.subject_classif_EC.y, train_size=.9, repetitions=10)

subj_n = [2, 5, 10, 15, 20, 25, 30]
# classification varying subjects and extracting features (very time consuming!) already done and saved in prova2.pickle (10 repetitions used)
datasetB = pickle.load(open("/home/andrea/Work/code/neuroconn/prova2.pickle", "rb"))
# redo the classification over subjects without extracting features using 100 repetitions for more reliable estimation
datasetB.subject_classif_EC.classify_over_subjects(n_subjects=subj_n, repetitions=100, extract_features=False)
# classify over subsets of sessions of different size to show how many
# sessions are needed to achieve good classification performance
# number of sessions need to be multiple of number of subjects to have a 
# balanced training and test set
#movie.subject_classif_EC.classify_over_sessions(n_sessions=[30, 60])

########################################################
# plot number of minimal features over number of subjects
n_feat_subj = np.zeros([len(subj_n), 10])  # 10 here depends on the number of repetitions used to extract features
for s, nsub in enumerate(subj_n):
    for r in range(10):
        n_feat_subj[s, r] = datasetB.subject_classif.features_extraction[nsub][r]['score'].shape[0] - 1
sns.set_palette('colorblind')
sns.set_style('darkgrid')
sns.set_context('talk')
fig = plt.figure()
plt.minorticks_on()
plt.fill_between(subj_n, n_feat_subj.mean(axis=1) + n_feat_subj.std(axis=1),
                 n_feat_subj.mean(axis=1) - n_feat_subj.std(axis=1), alpha=0.5, color='grey')
plt.scatter(subj_n, n_feat_subj.mean(axis=1), label="data (mean)", color='grey')
plt.xlabel('# subjects', fontsize=22)
plt.ylabel('minimal features', fontsize=22)
plt.yscale('log')
plt.xscale('log')

def fit_curves(xx, yy, extrapolate=None):
    # fit minimal features over subjects with different functions
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
    if extrapolate==None:
        extrapolate = xx.max()
    xx2 = np.linspace(xx.min(), extrapolate, 100)
    plt.plot(xx2, lin_f(xx2, popt_lin[0], popt_lin[1]), label=r"$a + b x$, SSE: %.4f" % (SSE_lin))
    plt.plot(xx2, log_f(xx2, popt_log[0], popt_log[1]), label=r"$a+ln(x) b$, SSE: %.4f" % (SSE_log))
    plt.plot(xx2, pow_f(xx2, popt_pow[0], popt_pow[1]), label=r"$a x^b$, SSE: %.4f" % (SSE_pow))
    plt.legend(fontsize=22)
    
fit_curves(xx=np.repeat(subj_n, 10), yy=n_feat_subj.flatten(), extrapolate=1000)
plt.grid(which='minor')

###############################################
#plot classification accuracy over number of subjects with curve fit and extrapolation to 1000 subjects
plt.figure()
plt.scatter(subj_n, datasetB.subject_classif_EC.score_over_subjects.mean(axis=1),
            color='grey', label="data (mean)")
plt.fill_between(subj_n,
                 np.percentile(datasetB.subject_classif_EC.score_over_subjects, 95, axis=1),
                 np.percentile(datasetB.subject_classif_EC.score_over_subjects, 5, axis=1),
                 alpha=0.5, color='grey')
plt.xlabel('# subjects', fontsize=22)
plt.ylabel('test-set accuracy', fontsize=22)
fit_curves(xx=np.repeat(subj_n, 100), 
           yy=datasetB.subject_classif_EC.score_over_subjects.flatten(), extrapolate=1000)
plt.xscale('log')
plt.grid(which='minor')
plt.ylim([.8, 1])
#######################################
# plot correlation of rankings for each repetition for a given number of subjects
subjN = 5
rr = np.zeros([10, len(datasetB.subject_classif.features_extraction[subjN][0]['ranking'])])
for r in range(10):
    rr[r,:] = datasetB.subject_classif.features_extraction[subjN][r]['ranking']
rm = np.corrcoef(rr)
rm[np.eye(10, dtype=bool)] = 0
plt.figure()
sns.heatmap(rm)
##############################




