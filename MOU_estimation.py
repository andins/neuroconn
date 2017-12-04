#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:50:53 2017

@author: andrea
"""

import numpy as np
import scipy.linalg as spl
import scipy.stats as stt
import matplotlib.pyplot as pp
from matplotlib.gridspec import GridSpec


def MOU_Lyapunov(ts_emp, SC_mask=None, norm_fc=None, true_S=None, true_C=None, verbose=0):
    """
    Estimation of MOU parameters (connectivity C, noise covariance Sigma,
    and time constant tau_x) with Lyapunov optimization as in: Gilson et al.
    Plos Computational Biology (2016).
    PARAMETERS:
        ts_emp: the timeseries data of the system to estimate, shape: T time points x P variables.
        SC_mask: mask of known non-zero values for connectivity matrix, for example 
        estimated by DTI
        norm_fc: normalization factor for FC. Normalization is needed to avoid high connectivity value that make
        the network activity explode. FC is normalized as FC *= 0.5/norm_fc. norm_fc can be specified to be for example
        the average over all entries of FC for all subjects or sessions in a given group. If not specified the normalization factor is
        the mean of 0-lag covariance matrix of the provided time series ts_emp. 
        true_S: true noise covariance (for testing of the algorithm on simulated data)
        true_C: true connectivity matrix (for testing of the algorithm on simulated data)
        verbose: verbosity level; 0: no output; 1: prints regularization parameters,
        estimated \tau_x, used lag \tau for lagged-covariance and evolution of the iterative
        optimization with final values of data covariance (Functional Connectivity) fitting;
        2: create a diagnostic graphics with cost function over iteration 
        (distance between true and estimated connectivity is also shown 
        if true_C is not None) and fitting of data covariance matrix (fitting of connectivity
        and Sigma are also shown if true_C and true_S are not None).
    RETURN:
        C: estimated connectivity matrix, shape [P, P] with null-diagonal
        Sigma: estimated noise covariance, shape [P, P]
        tau_x: estimated time constant (scalar)
        d_fit: a dictionary with diagnostics of the fit; keys are: iterations, distance and correlation
    """
    # TODO: look into regularization
    # TODO: make better graphics (deal with axes separation, etc.)
    # FIXME: tau_x in Matt origina script is calculated as the mean tau_x over sessions for each subject: why? Is this import? 

    
    # optimzation steps and rate
    n_opt = 10000
    epsilon_EC = 0.0005
    epsilon_Sigma = 0.05
    regul_EC = 0  # 0.5
    regul_Sigma = 0  # 0.001
    
    if verbose>0:
        print('regularization:', regul_EC, ';', regul_Sigma)
    
    N = ts_emp.shape[1] # number of ROIs
    
    n_tau = 3 # number of time shifts for FC_emp
    v_tau = np.arange(n_tau)
    i_tau_opt = 1 # time shift for optimization
    
    min_val_EC = 0. # minimal value for EC
    max_val_EC = 0.4 # maximal value for EC
    min_val_Sigma_diag = 0. # minimal value for Sigma
  
    
    # FC matrix (ts_emp matrix with stucture time x ROI index)
    n_T = ts_emp.shape[0]  # number of time samples
    ts_emp -= np.outer(np.ones(n_T),ts_emp.mean(0))
    FC_emp = np.zeros([n_tau,N,N])
    for i_tau in range(n_tau):
    	FC_emp[i_tau,:,:] = np.tensordot(ts_emp[0:n_T-n_tau,:],ts_emp[i_tau:n_T-n_tau+i_tau,:],axes=(0,0)) / float(n_T-n_tau-1)

    # normalize covariances (to ensure the system does not explode)
    if norm_fc is None:
        norm_fc = FC_emp[0,:,:].mean()
    FC_emp *= 0.5/norm_fc
    if verbose>0:
        print('max FC value (most of the distribution should be between 0 and 1):', FC_emp.mean())

    # autocovariance time constant
    log_ac = np.log(np.maximum(FC_emp.diagonal(axis1=1,axis2=2),1e-10))
    lin_reg = np.polyfit(np.repeat(v_tau,N),log_ac.reshape(-1),1)
    tau_x = -1./lin_reg[0]
    if verbose>0:
        print('inverse of negative slope (time constant):', tau_x)
    
    
    # mask for existing connections for EC and Sigma
    mask_diag = np.eye(N,dtype=bool)
    if SC_mask is None:
        mask_EC = np.logical_not(mask_diag) # all possible connections except self
    else:
        mask_EC = SC_mask
    mask_Sigma = np.eye(N,dtype=bool) # independent noise
    #mask_Sigma = np.ones([N,N],dtype=bool) # coloured noise
    
    # optimization
    if verbose>0:
        print('*opt*')
        print('i tau opt:', i_tau_opt)
    tau = v_tau[i_tau_opt]
    
    # objective FC matrices (empirical)
    FC0_obj = FC_emp[0,:,:]
    FCtau_obj = FC_emp[i_tau_opt,:,:]
    
    coef_0 = np.sqrt(np.sum(FCtau_obj**2)) / (np.sqrt(np.sum(FC0_obj**2))+np.sqrt(np.sum(FCtau_obj**2)))
    coef_tau = 1. - coef_0
    
    # initial network parameters
    EC = np.zeros([N,N])
    Sigma = np.eye(N)  # initial noise
    
    # best distance between model and empirical data
    best_dist = 1e10
    best_Pearson = 0.
    
    # record model parameters and outputs
    dist_FC_hist = np.zeros([n_opt])*np.nan # FC error = matrix distance
    Pearson_FC_hist = np.zeros([n_opt])*np.nan # Pearson corr model/objective
    dist_EC_hist = np.zeros([n_opt])*np.nan # FC error = matrix distance
    Pearson_EC_hist = np.zeros([n_opt])*np.nan # Pearson corr model/objective
    
    d_fit = dict()  # a dictionary to store the diagnostics of fit
    stop_opt = False
    i_opt = 0
    while not stop_opt:
        
        # calculate Jacobian of dynamical system
        J = -np.eye(N)/tau_x + EC
        		
        # calculate FC0 and FCtau for model
        FC0 = spl.solve_lyapunov(J,-Sigma)
        FCtau = np.dot(FC0,spl.expm(J.T*tau))
        
        # calculate error between model and empirical data for FC0 and FC_tau (matrix distance)
        err_FC0 = np.sqrt(np.sum((FC0-FC0_obj)**2))/np.sqrt(np.sum(FC0_obj**2))
        err_FCtau = np.sqrt(np.sum((FCtau-FCtau_obj)**2))/np.sqrt(np.sum(FCtau_obj**2))
        dist_FC_hist[i_opt] = 0.5*(err_FC0+err_FCtau)
        if not(true_C is None):
            dist_EC_hist[i_opt] = np.sqrt(np.sum((EC-true_C)**2))/np.sqrt(np.sum(true_C**2))
        	
        # calculate Pearson corr between model and empirical data for FC0 and FC_tau
        Pearson_FC_hist[i_opt] = 0.5*(stt.pearsonr(FC0.reshape(-1),FC0_obj.reshape(-1))[0]+stt.pearsonr(FCtau.reshape(-1),FCtau_obj.reshape(-1))[0])
        if not(true_C is None):
            Pearson_EC_hist[i_opt] = stt.pearsonr(EC.reshape(-1), true_C.reshape(-1))[0]
        
        # best fit given by best Pearson correlation coefficient for both FC0 and FCtau (better than matrix distance)
        if dist_FC_hist[i_opt]<best_dist:
            	best_dist = dist_FC_hist[i_opt]
            	best_Pearson = Pearson_FC_hist[i_opt]
            	i_best = i_opt
            	EC_best = np.array(EC)
            	Sigma_best = np.array(Sigma)
            	FC0_best = np.array(FC0)
            	FCtau_best = np.array(FCtau)
        else:
            stop_opt = i_opt>100
        
        # Jacobian update with weighted FC updates depending on respective error
        Delta_FC0 = (FC0_obj-FC0)*coef_0
        Delta_FCtau = (FCtau_obj-FCtau)*coef_tau
        Delta_J = np.dot(np.linalg.pinv(FC0),Delta_FC0+np.dot(Delta_FCtau,spl.expm(-J.T*tau))).T/tau
        # update conectivity and noise
        EC[mask_EC] += epsilon_EC * (Delta_J - regul_EC*EC)[mask_EC]
        EC[mask_EC] = np.clip(EC[mask_EC],min_val_EC,max_val_EC)
        
        Sigma[mask_Sigma] += epsilon_Sigma * (-np.dot(J,Delta_FC0)-np.dot(Delta_FC0,J.T) - regul_Sigma)[mask_Sigma]
        Sigma[mask_diag] = np.maximum(Sigma[mask_diag],min_val_Sigma_diag)
        
        # check if end optimization: if FC error becomes too large
        if stop_opt or i_opt==n_opt-1:
            stop_opt = True
            d_fit['iterations'] = i_opt
            d_fit['distance'] = best_dist
            d_fit['correlation'] = best_Pearson
            if verbose>0:
                print('stop at step', i_opt, 'with best dist', best_dist, ';best FC Pearson:', best_Pearson)
        else:
            if (i_opt)%20==0 and verbose>0:
                print('opt step:', i_opt)
                print('current dist FC:', dist_FC_hist[i_opt], '; current Pearson FC:', Pearson_FC_hist[i_opt])
            i_opt += 1
     
     		
        
        
    if verbose>1:
        # plots
        
        mask_nodiag = np.logical_not(np.eye(N,dtype=bool))
        mask_nodiag_and_not_EC = np.logical_and(mask_nodiag,np.logical_not(mask_EC))
        mask_nodiag_and_EC = np.logical_and(mask_nodiag,mask_EC)
        
        pp.figure()
        gs = GridSpec(2, 3)
            
        if not(true_C is None):
            pp.subplot(gs[0,2])
            pp.scatter(true_C, EC_best, marker='x')
            pp.xlabel('original EC')
            pp.ylabel('estimated EC')
            pp.text(pp.xlim()[0]+.05, pp.ylim()[1]-.05,
                     r'$\rho$: ' + str(stt.pearsonr(true_C[mask_EC], EC_best[mask_EC])[0]))
            
            
        if not(true_S is None):
            pp.subplot(gs[1,2])
            pp.scatter(true_S, Sigma_best,marker='x')
            pp.xlabel('original Sigma')
            pp.ylabel('estimated Sigma')
            pp.text(pp.xlim()[0]+.05, pp.ylim()[1]-.05,
                     r'$\rho_{diag}$: ' + str(stt.pearsonr(true_S.diagonal(), Sigma_best.diagonal())[0])
                     + r'$\rho_{off-diag}$: ' + str(stt.pearsonr(true_S[mask_nodiag], Sigma_best[mask_nodiag])[0])
                     )
        
            
        pp.subplot(gs[0,0:2])
        pp.plot(range(n_opt),dist_FC_hist, label='distance FC')
        pp.plot(range(n_opt),Pearson_FC_hist, label=r'$\rho$ FC')
        if not(true_C is None):
            pp.plot(range(n_opt),dist_EC_hist, label='distance EC')
            pp.plot(range(n_opt),Pearson_EC_hist, label=r'$\rho$ EC')
        pp.legend()
        pp.xlabel('optimization step')
        pp.ylabel('FC error')
        
            
        pp.subplot(gs[1,0])
        pp.scatter(FC0_obj[mask_nodiag_and_not_EC], FC0_best[mask_nodiag_and_not_EC], marker='x', color='k', label='not(SC)')
        pp.scatter(FC0_obj[mask_nodiag_and_EC], FC0_best[mask_nodiag_and_EC], marker='.', color='b', label='SC')
        pp.scatter(FC0_obj.diagonal(), FC0_best.diagonal(), marker= '.', color='c', label='diagonal')
        pp.legend()
        pp.xlabel('FC0 emp')
        pp.ylabel('FC0 model')
        
        
        pp.subplot(gs[1,1])
        pp.scatter(FCtau_obj[mask_nodiag_and_not_EC], FCtau_best[mask_nodiag_and_not_EC], marker='x', color='k', label='not(SC)')
        pp.scatter(FCtau_obj[mask_nodiag_and_EC], FCtau_best[mask_nodiag_and_EC], marker='.', color='b', label='SC')
        pp.scatter(FCtau_obj.diagonal(), FCtau_best.diagonal(), marker= '.', color='c', label='diagonal')
        pp.xlabel('FCtau emp')
        pp.ylabel('FCtau model')
        
    
    return EC_best, Sigma_best, tau_x, d_fit
