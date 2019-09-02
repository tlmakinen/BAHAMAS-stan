'''
Function library for computing aspects needed to sample BAHAMAS posterior
via Gibbs Sampling, first outlined by Shariff et al (2016) 
https://arxiv.org/pdf/1510.05954.pdf
'''
# modules we need for the analysis
import numpy as np
import pandas as pd
import scipy.stats
import scipy.linalg, scipy.special
# turn off annoying write warnings in pandas
pd.options.mode.chained_assignment = None  # default='warn'

# import bahamas modules
import bahamas
from bahamas import cosmology
from bahamas import vanilla_log_likelihood as vanilla
import priors

# Begin Gibbs Sampling Library
#---------------------------------------------------------------

# block-diagonal matrix for alpha, beta
def codeforA(ndat, alpha, beta): 
    A_i = np.matrix([[1, 0, 0], [0, 1, 0], [beta, -alpha, 1]])
    return scipy.linalg.block_diag(*([A_i,]*ndat))

# operations with A
def times_A_from_right(X, alpha, beta):
    X_times_A = np.copy(X)
    X_times_A[:,::3] += beta * X[:,2::3]
    X_times_A[:,1::3] -= alpha * X[:,2::3]
    return X_times_A

def times_Atranspose_from_left(X, alpha, beta):
    Atranspose_times_X = np.copy(X)
    Atranspose_times_X[::3] += beta*X[2::3]
    Atranspose_times_X[1::3] -= alpha*X[2::3]
    return Atranspose_times_X

# Population variance-covariance matrix of latents in D vector
# TODO: Toggle for nonsymmetric color distribution?
def codeforsigmaPinv(ndat, sigma_res, rc, rx):
    Sinv = np.diag([1 / rc**2, 1 / rx**2, 1 / sigma_res**2])
    return scipy.linalg.block_diag(*([Sinv,]*ndat))

# 3x3 prior matrix for LATENT quantities in D
def codeforsigmaDstar(cstar, xstar, mstar):
    return np.matrix([[cstar], [xstar], [mstar]])

# Sigma_A matrix: combines observed variances in sigmaCinv
# (uncertainties) and population-level latent variance-covariances
def codeforsigmaAinv(sigmaCinv, sigmaPinv, alpha, beta):
    sigmaC_times_A = times_A_from_right(sigmaCinv, alpha, beta)
    return times_Atranspose_from_left(sigmaC_times_A, alpha, beta) + sigmaPinv

# for prior distributions
def log_invgamma(x, a, b):
    if x < 1e-3:
        return -1e90
    return np.log(b**a / scipy.special.gamma(a) * x**(-a - 1) * np.exp(-b /x))


# prior cube of ndim, each dimension a uniform distrubution, U(0,1).
def makePriorCube(ndim):
    cube = []
    for i in range(ndim):
        cube.append(np.random.uniform(0,1))
    return  cube


# Create proposal gaussian distribution for MCMC steps, with constraints from priors
def gaussianProposal(mean, cov):
    return np.random.multivariate_normal(mean=mean, cov=cov)


#--------------------------------------------------------------------------------
#
# Gibbs Posterior Class Object
#
#---------------------------------------------------------------------------------
''' Here we store a bunch of useful attributes of our posterior BAHAMAS model so that
    we don't have to recompute them every time. Our model takes in the data (which doesn't change)
    and changes its attributes depending on the parameters that we change each step of the way 
    in our sampling process.
    
    Parameters to sample:
    
    parameters = ['alpha', 'beta', 
                  'rx', 'rc', 'sigma_res',
                  'cstar', 'xstar', 'mstar',
                  'omegam', 'w', 'h']
                  
    D =  {M_1,x1_1,c_1 ... M_n, x1_n, c_n}^T      column stack vector of latent vars
    D* = {Mstar, xstar, cstar}^T                  vector of population means of latent vars
    D** = {-19.3, 0, 0}^T                         vector of PRIOR means of D* quantities

    
'''
class posteriorModel(object):
    def __init__(self, J, sigmaCinv, log_sigmaCinv, data, ndat):
        self.J = J
        self.sigmaCinv = sigmaCinv
        self.log_sigmaCinv = log_sigmaCinv
        self.data = data
        self.ndat = ndat
        
        
    def latent_attributes(self, param):
        cosmo_param = param[8:]
        param = param[:8]
        Zcmb, Zhel, c, x1, mb = self.data.T

        ndat = self.ndat
        
        # compute mu from the data and cosmo_param
        Zcmb, Zhel = self.data.T[0:2]
        phi = self.data[:, 2:5]    
        mu = cosmology.muz(cosmo_param, Zcmb, Zhel)   # data extracted from self, only dependent on cosmo
        if np.any(np.isnan(mu)): # quit if hubble integral is not integrable
            return -np.inf

      
        J = self.J

        # extract data uncertainties
        sigmaCinv = self.sigmaCinv

        # extract parameters from argument--these are the ones we're going to infer
        alpha, beta, rx, rc, sigma_res = param[:5]
        cstar, xstar, mstar = param[5:8]    # population means of color, stretch, intrinsic magnitude (LATENTS)

        sigmaPinv = codeforsigmaPinv(ndat, sigma_res, rc, rx)


        # put matrices together       

        sigmaAinv = codeforsigmaAinv(sigmaCinv, sigmaPinv, alpha, beta)
        sigmaA = np.linalg.inv(sigmaAinv)
        
        # ---- Vectors ---- #
        # data vector D0
        X0 = [] # vector of observed values
        for i in range(ndat):
            X0.append(c[i])
            X0.append(x1[i])
            X0.append(mb[i] - mu[i])
 
        # population means D*
        b = np.matrix([[cstar], [xstar], [mstar]]) # Dstar
        Ystar = J * b  # J * Dstar


        # Matrix-multiply our data vector D0 with our covariance matrix
        sigmaCinv_X0 = np.matrix(np.einsum('ij,j', sigmaCinv, X0)).T # dot sigmaC and D0
        Delta = times_Atranspose_from_left(sigmaCinv_X0, alpha, beta)


        # Put together our sigmaDstar matrix of PRIOR variance-covariances of population-level latents in sigmaDstar
        sigma0 = np.diag([1**2, 10**2, 2**2])
        sigma0inv = np.linalg.inv(sigma0)

        # vincent code version
        Delta = times_Atranspose_from_left(sigmaCinv_X0, alpha, beta)


        cho_factorized_sigmaAinv = scipy.linalg.cho_factor(sigmaAinv, lower=True)
        muA = np.matrix(scipy.linalg.cho_solve(cho_factorized_sigmaAinv, Delta + sigmaPinv * Ystar))  # muA
          


        sigmakinv = - np.einsum('ij,ik,kl',J,sigmaPinv,np.linalg.solve(
                 sigmaAinv,
                 np.dot(sigmaPinv,J)
                )) \
                    + ndat*sigmaPinv[:3,:3] + sigma0inv

        sigmak = np.linalg.inv(sigmakinv)

        # D** vectors of PRIOR means of Dstar values
        bm = np.array([0,0,-19.3])

        sigmaA_times_Delta = np.linalg.solve(sigmaAinv,Delta)
        kstar = np.dot(
            sigmak, np.einsum('ji,jk,k...',J,sigmaPinv,sigmaA_times_Delta)[0] 
                    + np.dot(sigma0inv,bm))

        return kstar, np.linalg.inv(sigmakinv), muA, sigmaA
            
   
    def log_likelihood(self, param):
        # Impose prior constraints:
        if priors.constraints(param) is True:
            return -np.inf
        
        cosmo_param = param[8:]
        param = param[:8]
        Zcmb, Zhel, c, x1, mb = self.data.T
        ndat = self.ndat
        
        # compute mu from the data and cosmo_param
        Zcmb, Zhel = self.data.T[0:2]
        phi = self.data[:, 2:5]    
        mu = cosmology.muz(cosmo_param, Zcmb, Zhel)   # data extracted from self, only dependent on cosmo
        if np.any(np.isnan(mu)): # quit if hubble integral is not integrable
            return -np.inf

        J = self.J

        # extract data uncertainties
        sigmaCinv = self.sigmaCinv

        # extract parameters from argument
        alpha, beta, rx, rc, sigma_res = param[:5]
        cstar, xstar, mstar = param[5:8]    # population means of color, stretch, intrinsic magnitude (LATENTS)
        
        # priors
        sigmacstar = 1.0
        sigmaxstar =10.0
        sigmamo =2.0
        Mm = -19.3

        sigma0inv = np.diag([1./(sigmacstar)**2,1./(sigmaxstar)**2,1./(sigmamo)**2])
        bm = np.array([0,0,Mm])
        
        sigmaPinv = codeforsigmaPinv(ndat, sigma_res, rc, rx)
        sigmaAinv = codeforsigmaAinv(sigmaCinv, sigmaPinv, alpha, beta)

        # ---- Vectors ---- #
        # data vector D0
        X0 = []   # vector of observed values
        for i in range(ndat):
            X0.append(c[i])
            X0.append(x1[i])
            X0.append(mb[i] - mu[i])

        sigmaCinv_X0 = np.matrix(np.einsum('ij,j', sigmaCinv, X0)).T # dot sigmaC and X0
        Delta = times_Atranspose_from_left(sigmaCinv_X0, alpha, beta)

        b = np.matrix([[cstar], [xstar], [mstar]])
        Ystar = J * b  # J * Dstar


        # Lower triangular factorized sigmaA
        cho_factorized_sigmaAinv = scipy.linalg.cho_factor(sigmaAinv, lower=True)
    
        Y0 = np.matrix(scipy.linalg.cho_solve(cho_factorized_sigmaAinv, Delta + sigmaPinv * Ystar))  # muA


        chi1 = np.einsum('i,ij,j', np.array(X0), sigmaCinv, np.array(X0))
        chi2 = Y0.T * sigmaAinv * Y0
        chi3 = Ystar.T * sigmaPinv * Ystar
        chi4 = 0 #np.einsum('i,ij,j',bm,sigma0inv,bm)
        chisquare =  chi1 - chi2 + chi3 + chi4
        chisquare = np.array(chisquare)[0,0]

        #logdetsigmaPinv = -2 * ndat * np.log(rc * rx * sigma_res)
        parta = self.log_sigmaCinv - 2 * ndat * np.log(rc * rx * sigma_res) - 2 * np.sum(np.log(cho_factorized_sigmaAinv[0].diagonal()))

        partb = 0 # np.linalg.slogdet(sigma0inv)[1]
        # addition of low z anchor
        lz = 0.01
        sigma_lz = 0.0135

        mu_sim = cosmology.muz([0.30, 0.7, 0.72], lz, lz)
        mu_fit = cosmology.muz(cosmo_param, lz, lz)
        anchor = -0.5 * ((mu_sim - mu_fit)**2 / sigma_lz**2) + 1 / (np.sqrt(2 * np.pi) * sigma_lz)

        # INVGAMMA(0.003,0.003) prior distribution on sigma_res^2
        res_prior = log_invgamma(sigma_res**2, 0.003, 0.003)
    
        return -0.5 * (chisquare - parta - partb + 3 * ndat * np.log(2 * np.pi)) + anchor + res_prior

    def log_like_selection(self, param):
        J = self.J
        sigmaCinv = self.sigmaCinv
        log_sigmaCinv = self.log_sigmaCinv
        data = self.data
        ndat = self.ndat

        # Impose prior constraints:
        if priors.constraints(param) is True:
            return -np.inf

        return bahamas.vincent_log_likelihood(J, sigmaCinv, log_sigmaCinv, param, data, ndat)

    def log_correction(self, param):
        J = self.J
        sigmaCinv = self.sigmaCinv
        log_sigmaCinv = self.log_sigmaCinv
        data = self.data
        ndat = self.ndat

        # Impose prior constraints:
        if priors.constraints(param) is True:
            return -np.inf

        return bahamas.vincent_log_integral(param, data, ndat)