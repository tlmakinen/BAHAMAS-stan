'''
Script to build Gibbs Sampler Algorithm for posterior sampling
'''
# python modules
import math
import scipy.stats as stats
import numpy as np
import pandas as pd
import csv
import time

# bahamas modules
import priors
import bahamas
from bahamas import gibbs_library
from bahamas import latent_plots
#*********************************************************
# Parameter Vector we're sampling for:
#
# param = [alpha, beta, Rx, Rc, sigma_res, 
#                   cstar,xstar, mstar, omegam, w, h]
# 
# true values = [0.14, 3.2, np.exp(0.560333), np.exp(-2.3171), 0.1, 
#                            -0.06, 0.0, -19.1, 0.3, -1., 0.7]
#---------------------------------------------------------
#
#  Gibb's Sampler Algorithm
#
#---------------------------------------------------------
# STEP 1
# sample cosmological params {Omegam, w, h}
def step_one(posterior_object_for_sample, loglike, param, ndim, cov_proposal, n_accept):
    # sample cosmo from uniform prior and see if it moves towards higher density using an MH step. 
    # If it improves our likelihood, keep the parameter

    # Evaluate likelihood with original parameters for comparison
    old_loglike = posterior_object_for_sample.log_like_selection(param)
    old_correction = posterior_object_for_sample.log_correction(param)
    cosmo_param = param[8:10]            # EDIT: just sample omegam, omegade
    # --------------------------------------------------
    # MH sampler STEP from posterior  
    param_candidate = param 
    # sample from multivariate gaussian within our prior restrictions      
    param_candidate[8:10] = list(np.random.multivariate_normal(mean=cosmo_param, cov=cov_proposal))

    # compute new log likelihood with param candidates
    new_loglike = posterior_object_for_sample.log_like_selection(param_candidate)
    # compute new log-correction for selection effects
    new_correction = posterior_object_for_sample.log_correction(param_candidate)

    # if outside prior bounds, re-sample cosmo params from proposal distribution
    while np.isneginf(new_loglike):
        param_candidate[8:10] = list(np.random.multivariate_normal(mean=cosmo_param, cov=cov_proposal))
        new_loglike = posterior_object_for_sample.log_like_selection(param_candidate)

    # compute ratio of log likelihoods and move on if approaching higher density
    ln_acc_ratio = (new_loglike - old_loglike)    
   
    # now accept or reject the x candidate. Do this by sampling from U(0,1)        
    u = np.random.uniform(low=0, high=1)

    if (ln_acc_ratio > np.log(u)): 
        n_accept += 1  # for computing acceptance fraction for cosmo params
        param = param_candidate
        loglike = new_loglike
        log_correction = new_correction
        
    else:
        param[8:10] = cosmo_param
        loglike = old_loglike
        log_correction = old_correction

    return param,n_accept,loglike,log_correction

# -------------------------------------------------------------
# STEP 2
def step_two(posterior_object_for_sample, loglike, param, ndim, cov_proposal, n_accept_B):

    #ndat = posterior_object_for_sample.ndat

    # compute old log likelihood before we fiddle with params
    old_loglike = posterior_object_for_sample.log_like_selection(param)   
    old_correction = posterior_object_for_sample.log_correction(param)
    B_param = param[:2]    

    param_candidate = param                        # copy in our current big parameter vector
    param_candidate[0:2] = list(np.random.multivariate_normal(B_param, cov_proposal))
    # if we fall outside the prior bounds (in gibbs_library.py), MH-draw again:
    new_loglike = posterior_object_for_sample.log_like_selection(param_candidate)
    new_correction = posterior_object_for_sample.log_correction(param_candidate)

    while np.isneginf(new_loglike):
        param_candidate[0:2] = list(np.random.multivariate_normal(mean=B_param, cov=cov_proposal))
        new_loglike = posterior_object_for_sample.log_likelihood(param_candidate)

    # define the log of the acceptance fraction (difference in numer - denom)    
    ln_acc_ratio = (new_loglike - old_loglike)

        
    # now accept or reject the x candidate. Do this by sampling from U(0,1)        
    u = np.random.uniform(low=0, high=1)
        
    if (ln_acc_ratio > np.log(u)): 
        #print('B candidate accepted')
        n_accept_B += 1
        # move on in sample space
        param = param_candidate
        loglike = new_loglike
        log_correction = new_correction

    else:
        #print('B candidate rejected')
        param[0:2] = B_param
        loglike = old_loglike
        log_correction = old_correction

    return param,n_accept_B,loglike,log_correction
        
#------------------------------------------------------------------------------
# STEP 3
# sample D, (latent-layer data) and Dstar (SNIa population means)
def step_three(posterior_object_for_sample, D, param, ndim):
    # get latent attributes from posterior object
    kstar,sigmak, muA,sigmaA = posterior_object_for_sample.latent_attributes(param)   
    # sample new Dstar vector
    param[5:8] = np.random.multivariate_normal(mean=(kstar), cov=sigmak)   # generate new population mean vector
    
    # using new Dstar vector, sample a new column-stacked D latent variable vector 
    __,__, muA,sigmaA = posterior_object_for_sample.latent_attributes(param) 
    # generate new sampled data vector
    D_new = np.random.multivariate_normal(mean=np.ravel(muA), cov=sigmaA)   
    return D_new,param    # now our latent parameters are a 1D array, ordered D = [c1, x1, m1, ..., cn, xn, mn]

#-------------------------------------------------------------------------------------
# STEP 4
# sample sigma_res
def step_four(posterior_object_for_sample, D, param, ndim):
    ndat = posterior_object_for_sample.ndat
    lamb = 0.003                     # prior parameter
    var = 0.0                        # sum up the square of the variance in each latent Mi from the mean, M_*
    for M in D[2::3]:
        var += (M - param[7])**2   # subtract mstar from each latent magnitude
    
    # sample sigma_res^2 from inverse gamma distribution
    sigma_res_sq = stats.invgamma.rvs(a=(ndat/2) + lamb, scale=(var/2) + lamb, size=1, random_state=None)[0]
    param[4] = np.sqrt(sigma_res_sq)
    return param
#----------------------------------------------------------
# STEP 5
# sample Rx
def step_five(posterior_object_for_sample, D, param, ndim):
    ndat = posterior_object_for_sample.ndat
    var = 0.0                        # sum up the square of the variance in each latent xi from the mean, x_*
    for x in D[1::3]:
        var += ((x - param[6])**2)   # subtract xstar from each latent magnitude

    # sample R_x^2 from inverse gamma distribution
    rx_sq_new = stats.invgamma.rvs(a=(ndat/2), scale=(var/2), size=1, random_state=None)[0]
    param[2] = np.sqrt(rx_sq_new)
    return param
#---------------------------------------------------------------------
# STEP 6
# sample Rc
def step_six(posterior_object_for_sample, D, param, ndim):   
    ndat = posterior_object_for_sample.ndat
    var = 0.0                        # sum up the square of the variance in each latent ci from the mean, c_*
    for c in D[0::3]:
        var += ((c - param[5])**2)   # subtract mstar from each latent magnitude
        
    # sample R_c^2 from inverse gamma distribution
    rc_sq_new = stats.invgamma.rvs(a=(ndat/2), scale=(var/2), size=1, random_state=None)[0]
    param[3] = np.sqrt(rc_sq_new)

    return param

#----------------------------------------------------------------------
#
#     Running the Gibbs
#
#---------------------------------------------------------------------
# Sampler runs for niters_burn iterations to gain an estimate of cosmo and SALT-II covariance

def runGibbs(prior, posterior_object_for_sample, ndim, niters, niters_burn, outdir, snapshot=False, datafname=None, diagnose=False):
    print('-------- BEGINNING BURN-IN CHAIN --------')
    print('burn in iterations: ', niters_burn)

    prior = prior
    ndat = posterior_object_for_sample.ndat
    # set up empty D column-stacked latent variable array
    D = []
    for i in range(ndat):
        D.append(np.zeros((1,3)))

    # empty chain for parameter vectors
    chains = []
    loglike = 0
    param = prior   #  start parameter vector as the prior vector
    n_accept_cosmo = 0.0
    n_accept_B = 0.0

    chain_tab = pd.DataFrame(columns=['alpha', 'beta', 'rx', 'rc', 'sigma_res', 'cstar', 'xstar', 'mstar', 'omegam', 'w', 'h'])
    entry_list = []

    # create positive semi-definite covariance matrix for proposal distributions
    from sklearn import datasets
    #cov_proposal_cosmo = datasets.make_spd_matrix(n_dim = 2, random_state=None) * 0.1  # EDIT: just sample omegam, omegade
    cov_proposal_cosmo = np.array([[5, -0.02],
                                    [-0.02, 1]]) * (1./80) * ((2.38**2)/2)  # estimated from previous inference
    # now compute proposal sigmas for SALT-2 params
    #cov_proposal_B = datasets.make_spd_matrix(n_dim = 2, random_state=None) * 0.1
    cov_proposal_B = np.array([[0.02, -0.023],
                                [-0.023, 0.2]]) * (2.38**2 / 2)  # estimated from inference
    
    # We run our initial chains to get an estimate of the covariance between cosmo_params and cov between B_params    
    for iter in range(niters_burn):

        param1,n_accept_cosmo,loglike,log_corr = step_one(posterior_object_for_sample, loglike, param, ndim, cov_proposal_cosmo, n_accept_cosmo)
        param2,n_accept_B,loglike,log_corr = step_two(posterior_object_for_sample, loglike, param1, ndim, cov_proposal_B, n_accept_B)
        D,param3 = step_three(posterior_object_for_sample, D, param2, ndim)               # update D
        param4 = step_four(posterior_object_for_sample, D, param3, ndim) 
        param5 = step_five(posterior_object_for_sample, D, param4, ndim)
        param6 = step_six(posterior_object_for_sample, D, param5, ndim)
        entry_list.append(pd.Series(param6, index=chain_tab.columns))   # move on in sample space
        param = param6

        #print(param)

    #print('proposal estimation acceptance fraction = ', n_accept_cosmo / niters)
    # add to chains dataframe
    chain_tab = chain_tab.append(entry_list, ignore_index=True)

    # compute proposal sigmas for C and B:
    omegam = chain_tab['omegam'].values
    w = chain_tab['w'].values
    h = chain_tab['h'].values

    alpha = chain_tab['alpha'].values
    beta = chain_tab['beta'].values

    cosmo_proposal_cov = np.cov(np.matrix([omegam, w]))   # EDIT: just sample omegam, omegade
    B_proposal_cov = np.cov(np.matrix([alpha, beta]))       
    
    print('cosmo proposal covariance: ', cosmo_proposal_cov)
    #print('B proposal cov: ', B_proposal_cov)

    print('-------- STARTING ACTUAL SAMPLER CHAIN -------- ')
    #print('gibbs sampling for {} iterations'.format(niters))
            # set up empty D column-stacked latent variable array
    D = []
    for i in range(ndat):
        D.append(np.zeros((1,3)))

    # empty chain for parameter vectors
    chains = []
    loglike = 0
    
    D_chain = [] # for making diagnostic plot for subset of SN1a
    # make prior cube
    cube = gibbs_library.makePriorCube(ndim)
    param = gibbs_library.vanillaPrior(cube)  # start with prior again

    # reset acceptance fraction
    n_accept_cosmo = 0.0
    n_accept_B = 0.0
    true_param = [0.13, 2.56, 
            1.0, 0.1, 0.1, 0., 0., -19.3, 0.3, 0.7, 0.72]
    param[2:8] = true_param[2:8]
    print('og param', param)
    

    # open write file in case the PBS job gets killed part way:
    fname = outdir + 'post_chains_in_prog.csv'
    with open(fname, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            
        for iter in range(niters):
            #t0 = time.time()
            param,n_accept_cosmo,loglike,log_correction = step_one(posterior_object_for_sample, loglike, param, ndim, cosmo_proposal_cov, n_accept_cosmo)
            #t1 = time.time()
            #print('step 1 comp time: ', t1-t0)
           # t0 = time.time()
            param,n_accept_B,loglike,log_correction = step_two(posterior_object_for_sample, loglike, param, ndim, cov_proposal=B_proposal_cov, n_accept_B=n_accept_B)
            #t1 = time.time()
           # print('step 2 comp time: ', t1-t0)
           # t0 = time.time()
            #param0 = param
            
            #D,param0 = step_three(posterior_object_for_sample, D, param0, ndim) # update D
           # t1 = time.time()
           # print('step 3 comp time: ', t1-t0)
            #param = step_four(posterior_object_for_sample, D, param, ndim) 
            #param = step_five(posterior_object_for_sample, D, param, ndim)
            #param = step_six(posterior_object_for_sample, D, param, ndim)
            #print(param)
            
            
            loglike = [loglike]
            param_loglike = param + loglike + [log_correction]
            chains.append(np.asarray(param_loglike))   # move on in sample space
            
            if iter % 10 == 0:
                print('current parameter vector \nand likelihood for iter {}: '.format(iter), param_loglike)
            if diagnose == True:
                # latent vals from chain
                c = D[0::3]
                x1 = D[1::3]
                mB = D[2::3]
                # take all SN1a
                #c = c[::10]
                #x1 = x1[::10]
                #mB = mB[::10]
                D_save = [] # thinned latent vector to be saved
                for i in range(len(c)):
                    D_save.append(c[i])
                    D_save.append(x1[i])
                    D_save.append(mB[i])
                
                D_chain.append(D_save) # add to trace chain
                

            wr.writerow(param_loglike)    # write the parameter step as we go.

            if snapshot == True:
                if iter == niters-1:
                    latent_plots.plot_attributes(D, param, iter)

    print('sampling complete: cosmo param acceptance fraction = ', n_accept_cosmo/niters)
    print('sampling complete: SALT-2 param acceptance fraction = ', n_accept_B/niters)
    fname = outdir + 'post_chains.csv'
    with open(fname, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for i in chains:
            wr.writerow(i)
    
    if diagnose == True:
        fname = outdir + 'D_latent.csv'
        # save chain output for diagnostic purposes
        with open(fname, 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for i in D_chain:
                wr.writerow(i)


    return n_accept_cosmo / niters, n_accept_B / niters

