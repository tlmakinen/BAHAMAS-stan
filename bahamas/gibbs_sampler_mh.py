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
#                   cstar,xstar, mstar, omegam, omegade, h]
# 
# true values = [0.14, 3.2, np.exp(0.560333), np.exp(-2.3171), 0.1, 
#                            -0.06, 0.0, -19.3, 0.3, -1., 0.7]
#---------------------------------------------------------
#
#  Gibb's Sampler Algorithm
#
#---------------------------------------------------------
# STEP 1
# sample cosmological params C = {Omegam, Omegade}

def step_one(posterior_object_for_sample, loglike, param, ndim, cov_proposal, n_accept):


    # Evaluate likelihood with original parameters for comparison
    old_loglike = loglike
    #old_correction = posterior_object_for_sample.log_correction(param)
    cosmo_param = param[8:10]            # EDIT: just sample omegam, omegade
    # --------------------------------------------------
    # MH sampler STEP from posterior  
    param_candidate = param 
    # sample from multivariate gaussian within our prior restrictions      
    param_candidate[8:10] = list(np.random.multivariate_normal(mean=cosmo_param, cov=cov_proposal))

    # compute new log likelihood with param candidates
    new_loglike = posterior_object_for_sample.log_like_selection(param_candidate)
    # compute new log-correction for selection effects
    #new_correction = posterior_object_for_sample.log_correction(param_candidate)

    # if outside prior bounds, re-sample cosmo params from proposal distribution
   #while np.isneginf(new_loglike):
    #    param_candidate[8:10] = list(np.random.multivariate_normal(mean=cosmo_param, cov=cov_proposal))
    #    new_loglike = posterior_object_for_sample.log_like_selection(param_candidate)

    # compute ratio of log likelihoods and move on if approaching higher density
    ln_acc_ratio = (new_loglike - old_loglike)    
   
    # now accept or reject the x candidate. Do this by sampling from U(0,1)        
    u = np.random.uniform(low=0, high=1)

    if (ln_acc_ratio > np.log(u)): 
        n_accept += 1                       # for computing acceptance fraction for cosmo params
        param = param_candidate
        loglike = new_loglike
        #log_correction = new_correction
        
    else:
        param[8:10] = cosmo_param
        loglike = old_loglike
        #log_correction = old_correction

    return param,n_accept,loglike #,log_correction

# -------------------------------------------------------------
# STEP 2
def step_two(posterior_object_for_sample, loglike, param, ndim, cov_proposal, n_accept_B):

    # compute old log likelihood before we fiddle with params
    old_loglike = loglike 
    #old_correction = posterior_object_for_sample.log_correction(param)
    B_param = param[:2]    

    param_candidate = param                        # copy in our current big parameter vector
    param_candidate[0:2] = list(np.random.multivariate_normal(B_param, cov_proposal))
    # if we fall outside the prior bounds (in gibbs_library.py), MH-draw again:
    new_loglike = posterior_object_for_sample.log_like_selection(param_candidate)
    #new_correction = posterior_object_for_sample.log_correction(param_candidate)

    #while np.isneginf(new_loglike):
    #    param_candidate[0:2] = list(np.random.multivariate_normal(mean=B_param, cov=cov_proposal))
    #    new_loglike = posterior_object_for_sample.log_likelihood(param_candidate)

    # define the log of the acceptance fraction (difference in numer - denom)    
    ln_acc_ratio = (new_loglike - old_loglike)

        
    # now accept or reject the x candidate. Do this by sampling from U(0,1)        
    u = np.random.uniform(low=0, high=1)
        
    if (ln_acc_ratio > np.log(u)): 
        n_accept_B += 1
        # move on in sample space
        param = param_candidate
        loglike = new_loglike
        #log_correction = new_correction

    else:
        param[0:2] = B_param
        loglike = old_loglike
        #log_correction = old_correction

    return param,n_accept_B,loglike #,log_correction
        
#------------------------------------------------------------------------------
# STEP 3
# sample D, (latent-layer data) and Dstar (SNIa population means) via MH
def step_three(posterior_object_for_sample, loglike, D, param, ndim, cov_proposal_p, n_accept_dstar):
    # get latent attributes from posterior object
    #kstar,sigmak, muA,sigmaA = posterior_object_for_sample.latent_attributes(param)  

    # old log-likelihood
    old_loglike = loglike

    param_candidate = param

    # sample new Dstar vector
    param_candidate[5:8] = np.random.multivariate_normal(mean=(param[5:8]), cov=cov_proposal_p)   # generate new population mean vector
    

    # MH-sample population mean vector (Dstar) using corrected posterior
    new_loglike = posterior_object_for_sample.log_like_selection(param_candidate)

    # ratio of proposed and old loglikes
    ln_acc_ratio = (new_loglike - old_loglike)
    
    u = np.random.uniform(low=0, high=1)

    
    # MH-draw
    if (ln_acc_ratio > np.log(u)): 
        n_accept_dstar += 1
        # move on in sample space
        param = param_candidate
        loglike = new_loglike


    else:
        param = param
        loglike = old_loglike


    # using MH-sampled Dstar vector, sample a new column-stacked D latent variable vector 
    __,__, muA,sigmaA = posterior_object_for_sample.latent_attributes(param) 
    # generate new sampled data vector
    D_new = np.random.multivariate_normal(mean=np.ravel(muA), cov=sigmaA)   

    return D_new,param,n_accept_dstar,loglike    # now our latent parameters are a 1D array, ordered D = [c1, x1, m1, ..., cn, xn, mn]

#-------------------------------------------------------------------------------------
# STEP 4
# sample sigma_res
def step_four(posterior_object_for_sample, loglike, D, param, ndim, cov_proposal_sr, n_accept_sigmares):

    param_candidate = param
    # compute old likelihood
    old_loglike = loglike

    ndat = posterior_object_for_sample.ndat
    lamb = 0.003                     # prior parameter
    var = 0.0                        # sum up the square of the variance in each latent Mi from the mean, M_*
    for M in D[2::3]:
        var += (M - param[7])**2   # subtract mstar from each latent magnitude

    
    # sample sigma_res^2 from inverse gamma distribution
    sigma_res =  np.random.normal(loc=(param[4]), scale=cov_proposal_sr)
    param_candidate[4] = sigma_res

    new_loglike = posterior_object_for_sample.log_like_selection(param_candidate)

    # MH-draw using corrected posterior
    ln_acc_ratio = new_loglike - old_loglike
    
    u = np.random.uniform(low=0, high=1) # random number

    # MH-draw
    if (ln_acc_ratio > np.log(u)): 
        n_accept_sigmares += 1
        # move on in sample space
        param = param_candidate
        loglike = new_loglike


    else:
        param = param


    return param,n_accept_sigmares,loglike
#----------------------------------------------------------
# STEP 5
# sample Rx ~INVGAMMA
def step_five(posterior_object_for_sample, loglike, D, param, ndim, cov_proposal_rx, n_accept_rx):
    old_loglike = loglike

    param_candidate = param

    ndat = posterior_object_for_sample.ndat
    var = 0.0                        # sum up the square of the variance in each latent xi from the mean, x_*
    #for x in D[1::3]:
    #    var += ((x - param[6])**2)   # subtract xstar from each latent magnitude


    # sample R_x^2 from inverse gamma distribution
    rx_new =  np.random.normal(loc=(param[2]), scale=cov_proposal_rx)
    #rx_sq_new =  stats.invgamma.rvs(a=(ndat/2), scale=(var/2), size=1, random_state=None)[0]
    param_candidate[2] = rx_new

    new_loglike = posterior_object_for_sample.log_like_selection(param_candidate)

    # MH-draw using corrected posterior
    ln_acc_ratio = new_loglike - old_loglike
    
    u = np.random.uniform(low=0, high=1) # random number

    # MH-draw
    if (ln_acc_ratio > np.log(u)): 
        n_accept_rx += 1
        # move on in sample space
        param = param_candidate
        loglike = new_loglike
    

    else:
        param = param

    return param,n_accept_rx,loglike
#---------------------------------------------------------------------
# STEP 6
# sample Rc ~INVGAMMA
def step_six(posterior_object_for_sample, loglike, D, param, ndim, cov_proposal_rc, n_accept_rc):   
    old_loglike = loglike

    param_candidate = param

    ndat = posterior_object_for_sample.ndat
    var = 0.0                        # sum up the square of the variance in each latent ci from the mean, c_*
    #for c in D[0::3]:
    #    var += ((c - param[5])**2)   # subtract mstar from each latent magnitude

    rc_new =  np.random.normal(loc=(param[3]), scale=cov_proposal_rc)
    param_candidate[3] = rc_new

    new_loglike = posterior_object_for_sample.log_like_selection(param_candidate)

    # MH-draw using corrected posterior
    ln_acc_ratio = new_loglike - old_loglike

    u = np.random.uniform(low=0, high=1) # random number

    # MH-draw
    if (ln_acc_ratio > np.log(u)): 
        n_accept_rc += 1
        # move on in sample space
        param = param_candidate
        loglike = new_loglike


    else:
        param = param

    return param,n_accept_rc,loglike

#----------------------------------------------------------------------
#
#     Running the Gibbs
#
#---------------------------------------------------------------------
# Sampler runs for niters_burn iterations to gain an estimate of cosmo and SALT-II covariance

def runGibbs(prior, posterior_object_for_sample, ndim, niters, outdir, snapshot=False, datafname=None, diagnose=False):
    ndat = posterior_object_for_sample.ndat
    # set up empty D column-stacked latent variable array
    D = []
    for i in range(ndat):
        D.append(np.zeros((1,3)))

    # empty chain for parameter vectors
    chains = []
    param = prior   #  start parameter vector as the prior vector

    # evaluate log-likelihood at the original parameter vector
    loglike = posterior_object_for_sample.log_like_selection(param)

    # define acceptance fractions
    n_accept_cosmo = 0.0      # step 1
    n_accept_B = 0.0          # step 2
    n_accept_dstar = 0.0      # step 3
    n_accept_sigmares = 0.0   # step 4   
    n_accept_rx = 0.0         # step 5
    n_accept_rc = 0.0         # step 6

    # define jump proposals
    cov_proposal_cosmo = np.array([[0.00862212, 0.01333633],
                                [0.01333633, 0.02501767]]) * ((2.38**2)/2) * 10 # EDIT: just sample omegam, omegade

    cov_proposal_B = np.array([[ 7.51338227e-05, -7.81496651e-07],
                             [-7.81496651e-07,  7.63401552e-03]]) * (2.38**2 / 2) * 10

    cov_proposal_p = np.array([[ 2.26136092e-05,  4.55322597e-06, -8.29879371e-06],
                              [ 4.55322597e-06,  2.29976669e-03,  3.91206305e-05],
                                 [-8.29879371e-06,  3.91206305e-05,  4.14424789e-04]]) * (2.38**2 / 3) * 10

    cov_proposal_sr = 0.00013154 * (2.38**2) * 10
    cov_proposal_rx = 0.00119617 * (2.38**2) * 10
    cov_proposal_rc = 1.27621525e-05 * (2.38**2) * 10



    print('-------- STARTING ACTUAL SAMPLER CHAIN -------- ')
    print('gibbs sampling for {} iterations'.format(niters))


    # empty chain for parameter vectors
    chains = []

    
    D_chain = [] # for making diagnostic plot for subset of SN1a
    # start with prior again


    true_param = [0.13, 2.56, 
            1.0, 0.1, 0.1, 0., 0., -19.3, 0.3, 0.7, 0.72]
    

    # open write file in case the PBS job gets killed part way:
    fname = outdir + 'post_chains_in_prog.csv'
    with open(fname, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            
            

        for iter in range(niters):

            param,n_accept_cosmo,loglike    = step_one(posterior_object_for_sample, loglike, param, ndim, cov_proposal_cosmo, n_accept_cosmo)
            param,n_accept_B,loglike        = step_two(posterior_object_for_sample, loglike, param, ndim, cov_proposal=cov_proposal_B, n_accept_B=n_accept_B)
            D,param,n_accept_dstar,loglike  = step_three(posterior_object_for_sample, loglike, D, param, ndim, cov_proposal_p, n_accept_dstar) # update D
            param,n_accept_sigmares,loglike = step_four(posterior_object_for_sample, loglike, D, param, ndim, cov_proposal_sr, n_accept_sigmares) 
            param,n_accept_rx,loglike       = step_five(posterior_object_for_sample, loglike, D, param, ndim, cov_proposal_rx, n_accept_rx)
            param,n_accept_rc,loglike       = step_six(posterior_object_for_sample, loglike, D, param, ndim, cov_proposal_rc, n_accept_rc)

            param_loglike = param + [loglike]
            chains.append(np.asarray(param_loglike))   # move on in sample space
            
            if iter % 1 == 0:
                print('current parameter vector \nand likelihood for iter {}: '.format(iter), param_loglike)
            if diagnose == True:
                # latent values from chain
                c = D[0::3]
                x1 = D[1::3]
                mB = D[2::3]
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

    acc_numbs =  [n_accept_cosmo, n_accept_B, n_accept_dstar, n_accept_sigmares, n_accept_rx, n_accept_rc]
    print('acc numbs: ', acc_numbs)
    acceptance_fracs = []
    acceptance_fracs = [i / niters for i in acc_numbs]

    print('sampling complete')
    for step in range(len(acc_numbs)):
        print('acceptance fraction for step {}: '.format(step + 1), acceptance_fracs[step])


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
   

    return acceptance_fracs

