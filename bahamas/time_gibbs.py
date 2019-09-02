'''
Script to build Gibbs Sampler Algorithm for posterior sampling
'''
# python modules
import math
import scipy.stats as stats
import numpy as np
import pandas as pd
import csv

# bahamas modules
import priors
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
def step_one(posterior_object_for_sample, D, loglike, param, ndim, cov_proposal, n_accept):
    # sample cosmo from uniform prior and see if it moves towards higher density using an MH step. 
    # If it improves our likelihood, keep the parameter

    # Evaluate likelihood with original parameters for comparison
    old_loglike = posterior_object_for_sample.log_likelihood(param)
    cosmo_param = param[8:10]            # EDIT: just sample omegam, omegade
    # --------------------------------------------------
    # MH sampler STEP from posterior  
    param_candidate = param 
    # sample from multivariate gaussian within our prior restrictions      
    param_candidate[8:10] = list(np.random.multivariate_normal(mean=cosmo_param, cov=cov_proposal))

    # compute new log likelihood with param candidates
    new_loglike = posterior_object_for_sample.log_likelihood(param_candidate)
    # if outside prior bounds, re-sample cosmo params from proposal distribution
    while new_loglike == 0:
        param_candidate[8:10] = list(np.random.multivariate_normal(mean=cosmo_param, cov=cov_proposal))
        new_loglike = posterior_object_for_sample.log_likelihood(param_candidate)

    # compute ratio of log likelihoods and move on if approaching higher density
    ln_acc_ratio = (new_loglike - old_loglike)    
   
    # now accept or reject the x candidate. Do this by sampling from U(0,1)        
    u = np.random.uniform(low=0, high=1)

    if (ln_acc_ratio > np.log(u)) or (ln_acc_ratio > 0.): 
        n_accept += 1  # for computing acceptance fraction for cosmo params
        param = param_candidate
        loglike = new_loglike
        
    else:
        param[8:10] = cosmo_param
        loglike = old_loglike

    return param,n_accept,loglike

# -------------------------------------------------------------
# STEP 2
def step_two(posterior_object_for_sample, D, loglike, param, ndim, cov_proposal, n_accept_B):

    #ndat = posterior_object_for_sample.ndat

    # compute old log likelihood before we fiddle with params
    old_loglike = posterior_object_for_sample.log_likelihood(param)   
    B_param = param[:2] 
   
    # MH sampler STEP from posterior
    r = np.random.uniform(low = 0, high = 1, size=2)


    param_candidate = param                        # copy in our current big parameter vector
    param_candidate[0:2] = list(np.random.multivariate_normal(B_param, cov_proposal))
    # if we fall outside the prior bounds (in gibbs_library.py), MH-draw again:
    new_loglike = posterior_object_for_sample.log_likelihood(param_candidate)

    while new_loglike == 0:
        param_candidate[0:2] = list(np.random.multivariate_normal(mean=B_param, cov=cov_proposal))
        new_loglike = posterior_object_for_sample.log_likelihood(param_candidate)

    # define the log of the acceptance fraction (difference in numer - denom)    
    ln_acc_ratio = (new_loglike - old_loglike)

        
    # now accept or reject the x candidate. Do this by sampling from U(0,1)        
    u = np.random.uniform(low=0, high=1)
        
    if (ln_acc_ratio > np.log(u)) or (ln_acc_ratio > 0.): 
        #print('B candidate accepted')
        n_accept_B += 1
        # move on in sample space
        param = param_candidate
        loglike = new_loglike

    else:
        #print('B candidate rejected')
        param[0:2] = B_param
        loglike = old_loglike

    return param,n_accept_B,loglike
        
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
                                    [-0.02, 1]]) * (1./80)
    # now compute proposal sigmas for SALT-2 params
    #cov_proposal_B = datasets.make_spd_matrix(n_dim = 2, random_state=None) * 0.1
    cov_proposal_B = np.array([[0.02, -0.023],
                                [-0.023, 0.2]])
    #print('B proposal sigmas: ', cov_proposal_B)
    # We run our initial chains to get an estimate of the covariance between cosmo_params and cov between B_params    
    for iter in range(niters_burn):

        param1,n_accept_cosmo,loglike = step_one(posterior_object_for_sample, D, loglike, param, ndim, cov_proposal_cosmo, n_accept_cosmo)
        param2,n_accept_B,loglike = step_two(posterior_object_for_sample, D, loglike, param1, ndim, cov_proposal_B, n_accept_B)
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
    

    # open write file in case the PBS job gets killed part way:
    fname = outdir + 'post_chains_in_prog.csv'
    with open(fname, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            
        for iter in range(niters):
            param,n_accept_cosmo,loglike = step_one(posterior_object_for_sample, D, loglike, param, ndim, cosmo_proposal_cov, n_accept_cosmo)
            param,n_accept_B,loglike = step_two(posterior_object_for_sample, D, loglike, param, ndim, cov_proposal=B_proposal_cov, n_accept_B=n_accept_B)
            D,param = step_three(posterior_object_for_sample, D, param, ndim) # update D
            param = step_four(posterior_object_for_sample, D, param, ndim) 
            param = step_five(posterior_object_for_sample, D, param, ndim)
            param = step_six(posterior_object_for_sample, D, param, ndim)

            
            loglike = [loglike]
            param_loglike = param + loglike
            chains.append(np.asarray(param_loglike))   # move on in sample space
            
            if iter % 100 == 0:
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
    fname = outdir + 'D_latent.csv'

    # save chain output for diagnostic purposes
    with open(fname, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for i in D_chain:
            wr.writerow(i)


    return n_accept_cosmo / niters, n_accept_B / niters



# --------------------------------------------
#
#  MPI-enabled Gibbs Sampling
#
#---------------------------------------------
# function to walk through steps of gibbs sampler
#def eval_Gibbs():
    

#def runGibbs_MPI(prior, posterior_object_for_sample, 
#            ndim, niters, niters_burn, outdir, snapshot=False, datafname=None, diagnose=False):

    #from mpi4py import MPI
    #comm = MPI.COMM_WORLD
    
    #size = comm.Get_size()
    #rank = comm.Get_rank()

    

# --------------------------------------------
#
#  Testing Gibbs Posterior Draws
#
#---------------------------------------------

def runGibbsFreeze(prior, param, posterior_object_for_sample, ndim, niters, niters_burn, outdir, plot=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # make prior cube
    cube = gibbs_library.makePriorCube(ndim)

    # prior cube for each parameter. Takes as input uniform prior cube
    #def makePrior(cube, ndim=1, nparams=1):
        #cube[0] = np.random.uniform(0, 1)                    # alpha
        #cube[1] = np.random.uniform(0, 4)                    # beta
        #cube[2] = priors.log_uniform(cube[2], 10**-5, 10**2) # Rx
        #cube[3] = priors.log_uniform(cube[3], 10**-5, 10**2) # Rc (CHANGE FOR NON-SYMMETRIC)
        #cube[4] = cube[4] * 1                                # sigma_res (in likelihood code)
        #cube[5] = priors.gaussian(cube[5], 0, 1**2)          # cstar 
        #cube[6] = priors.gaussian(cube[6], .0796, .02)       # xstar
        #cube[7] = priors.gaussian(cube[7], -19.3, 2.)        # mstar
        #cube[8] = cube[8] * 1                                # omegam
        #cube[9] = cube[9] * -4                               # w   EDIT
        #cube[10] = 0.3 + cube[10] * 0.7                      # h   EDIT
        #return cube
        # start off params with priors
    
    true_param = param
    #true_param =  [.14, 3.2, np.exp(.560333), np.exp(-2.3171), .1, -1.738, 0.0796, -2.605, .3, 0.7, .7]

    #true_param = [0.13, 2.56, 1.0, 0.1, 0.1, 0, 0, -19.3, 0.3, 0.7, 0.72]
    param = true_param
    
    ndat = posterior_object_for_sample.ndat
    # set up empty D column-stacked latent variable array
    D = []
    for i in range(ndat):
        D.append(np.zeros((1,3)))
    # empty chain for parameter vectors
    chains = []
    entry_list = []

    param[2:8] = prior[2:8]   #  start parameter vector as the prior vector, freezing cosmo and SALT-2
    #param[8:] = prior[8:]

    #print(param)
    n_accept_cosmo = 0.0
    n_accept_B = 0.0

    ## TEST BURN-IN ##
        # empty chain for parameter vectors
    chains = []
    
    print('initial parameter vector = ', param)
    
    n_accept_cosmo = 0.0
    n_accept_B = 0.0

    chain_tab = pd.DataFrame(columns=['alpha', 'beta', 'rx', 'rc', 'sigma_res', 'cstar', 'xstar', 'mstar', 'omegam', 'w', 'h'])
    entry_list = []

    # create positive semi-definite covariance matrix for proposal distributions
    from sklearn import datasets
    cov_proposal_cosmo = datasets.make_spd_matrix(n_dim = 3, random_state=None) * 0.8
    # now compute proposal sigmas for SALT-2 params
    cov_proposal_B = datasets.make_spd_matrix(n_dim = 2, random_state=None) * 0.8

    # We run our initial chains to get an estimate of the covariance between cosmo_params and cov between B_params    
    #for iter in range(niters_burn):
        #param,n_accept_cosmo = step_one(posterior_object_for_sample, D, param, ndim, cov_proposal_cosmo, n_accept_cosmo)
        #param,n_accept_B = step_two(posterior_object_for_sample, D, param, ndim, cov_proposal_B, n_accept_B)
        #entry_list.append(pd.Series(param, index=chain_tab.columns))   # move on in sample space
        #print(param)

    #print('proposal estimation acceptance fraction = ', n_accept_cosmo / niters)
    # add to chains dataframe
    #chain_tab = chain_tab.append(entry_list, ignore_index=True)

    # compute proposal sigmas for C and B:
    #omegam = chain_tab['omegam'].values
    #w = chain_tab['w'].values
    #h = chain_tab['h'].values

    #alpha = chain_tab['alpha'].values
    #beta = chain_tab['beta'].values

    #cov_proposal_cosmo = np.cov(np.matrix([omegam, w, h]))  # estimated proposal covariance
    #cov_proposal_B = np.cov(np.matrix([alpha, beta]))  

    ##



    print('--------- STARTING ACTUAL SAMPLER CHAIN ---------- ')
            # set up empty D column-stacked latent variable array
    D = []
    for i in range(ndat):
        D.append(np.zeros((1,3)))

    # empty chain for parameter vectors
    chains = []


    # open write file in case the PBS job gets killed part way:
    fname = outdir + 'post_chains_in_prog.csv'
    with open(fname, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

            
        for iter in range(niters):
            #param,n_accept_cosmo = step_one(posterior_object_for_sample, D, param, ndim, cov_proposal_cosmo, n_accept_cosmo)
            #param,n_accept_B = step_two(posterior_object_for_sample, D, param, ndim, cov_proposal=cov_proposal_B, n_accept_B=n_accept_B)
            D,param = step_three(posterior_object_for_sample, D, param, ndim) # update D
            param = step_four(posterior_object_for_sample, D, param, ndim) 
            param = step_five(posterior_object_for_sample, D, param, ndim)
            param = step_six(posterior_object_for_sample, D, param, ndim)

            if plot == True:

                D_arr = []
                for i in range(len(D[1::3])):
                    if (i % 3 == 0):
                        D_arr.append([D[i], D[i+1], D[i+2]])   # append in order: [c, x, m]
            
                mstar_latent = np.mean(D[2::3])          # get latent cstar
                sigma_res_latent = np.std(D[2::3])              # get latent rx
                # plot histograms
                if iter in [0, 1, 2, 3, 4, 5, 6, 7, 8, 100, 200]:

                    fig,ax = plt.subplots(1, 1, figsize = (6,6))

                    ax.hist(D[2::3], bins=50, density=True)
                    ax.set_xlabel(r'data ${m_B}$', fontsize=15)
                    ax.set_xlabel('latent mB distribution for iter {}'.format(iter), fontsize=15)
                    s = 'latent mstar for iter {} = '.format(iter) + '{0:8.3f}'.format(mstar_latent)
                    s += '\n latent sigma_res for iter {} = '.format(iter) + '{0:8.3f}'.format(sigma_res_latent)
                    s += '\n current value for mstar in chain = {0:8.3f}'.format(param[7])
                    s +='\n '
                    s += r'current value for sigma_res in chain = {0:8.3f}'.format(param[4])

                    m_hist = np.histogram(D[2::3], bins=50, density=True)
                    x_pos = np.median(m_hist[1])
                    y_pos = 0.3*np.max(m_hist[0])
                    #print(y_pos)
                    ax.text(-18, y_pos, s)
                    plt.savefig('latent_mB_{}.png'.format(iter))
            
            if iter % 100 == 0:
                print('current parameter vector for iter {}: '.format(iter), param)

            chains.append(np.asarray(param))   # move on in sample space

            wr.writerow(param)    # write the parameter step as we go.


    print('sampling complete: cosmo param acceptance fraction = ', n_accept_cosmo/niters)
    print('sampling complete: SALT-2 param acceptance fraction = ', n_accept_B/niters)
    fname = outdir + 'post_chains.csv'
    with open(fname, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for i in chains:
            wr.writerow(i)