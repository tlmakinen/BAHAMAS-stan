#import bahamas.cosmology as cosmology
import bahamas.cosmology as cosmology
import numpy as np
import scipy.linalg, scipy.special

# code adapted from vincent's selection effects work in 2017

def codeforA(ndat, alpha, beta): 
    A_i = np.matrix([[1, 0, 0], [0, 1, 0], [beta, -alpha, 1]])
    return scipy.linalg.block_diag(*([A_i,]*ndat))

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

# prior variance-covariance matrix for Dstar population means
def codeforsigmaPinv(ndat, sigma_res, rc, rx):
    Sinv = np.diag([1 / rc**2, 1 / rx**2, 1 / sigma_res**2])
    return scipy.linalg.block_diag(*([Sinv,]*ndat))

def log_invgamma(x, a, b):
    if x < 1e-3:
        return -1e90
    return np.log(b**a / scipy.special.gamma(a) * x**(-a - 1) * np.exp(-b /x))

def log_likelihood(J, sigmaCinv, log_sigmaCinv, param, cosmo_param, data, mu, ndat):
    Zcmb, Zhel, c, x1, mb = data.T

    alpha, beta, rx, rc, sigma_res = param[:5]
    cstar, xstar, mstar = param[5:8]
    
    # matrices - block diagonal
    A =  codeforA(ndat, alpha, beta)
    sigmaPinv = codeforsigmaPinv(ndat, sigma_res, rc, rx)
    sigmaCinv_A = times_A_from_right(sigmaCinv, alpha, beta)

    # matrices - combinations
    sigmaAinv = times_Atranspose_from_left(sigmaCinv_A, alpha, beta) + sigmaPinv

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
    chisquare =  chi1 - chi2 + chi3
    chisquare = np.array(chisquare)[0,0]
 
    
    logdetsigmaPinv = -2 * ndat * np.log(rc * rx * sigma_res)
    parta = log_sigmaCinv - 2 * ndat * np.log(rc * rx * sigma_res) - 2 * np.sum(np.log(cho_factorized_sigmaAinv[0].diagonal()))

    # addition of low z anchor
    lz = 0.01
    sigma_lz = 0.0135

    mu_sim = cosmology.muz([0.30, 0.7, 0.72], lz, lz)
    mu_fit = cosmology.muz(cosmo_param, lz, lz)
    anchor = -0.5 * ((mu_sim - mu_fit)**2 / sigma_lz**2) + 1 / (np.sqrt(2 * np.pi) * sigma_lz)

    # EDIT: remove anchor for now
    #anchor = 0.0
    
    # INVGAMMA(0.003,0.003) prior distribution on sigma_res^2
    res_prior = log_invgamma(sigma_res**2, 0.003, 0.003)
    
    return -0.5 * (chisquare - parta + 3 * ndat * np.log(2 * np.pi)) + anchor + res_prior
