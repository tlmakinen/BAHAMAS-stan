'''
Here we present corrections to the vanilla BAHAMAs log likelihood
to account for selection effects (see Kelly et al. 2007 for an
introduction).

We focus on comparing two models:
- Rubin (see more in Rubin et al. 2015)
- Vincent (derived by Vincent Chen 2017)

Notation is introduced in March et al. 2011 and Rubin et al. 2015
'''

import numpy as np
import pandas as pd
import scipy.stats
import bahamas.cosmology as cosmology

'''
log_indiv_selection_fn = log(P(I_i | Phi_i, Psi))

Assume the selection function for a given set of c_i, x_i, and mB_i 
is a logistic function = 1 / 1 + exp(- (gc * c_i + gx * x_i + gm * mB_i + eps))
- Found gammas and epsilon via a logistic regression on the full dataset (Dylan Jow 2017)

# TODO: Find coeffiicents via a probit regression
# old snls coeffs: -1.7380184749999987, 0.07955165160000005, -2.60483735, 62.15588654

'''
# trained on DES sims
eps,gmB,gc,gx1 = (29.96879778, -1.34334963,  0.45895811,  0.06703621)

#gc,gx1,gmB,eps = (-1.7380184749999987, 0.07955165160000005, -2.60483735, 62.15588654)


def log_indiv_selection_fn(phi_i, selection_param=(gc, gx1, gmB, eps)):
    coefs = np.array(selection_param)
    position = np.array([*phi_i, 1])
    argument = np.dot(coefs, position)
    return scipy.stats.norm.logcdf(np.sqrt(np.pi/8)*argument) # must be a logcdf so it dies/grows to 0/1 at the right speed 

'''
log_latent_marginalized_indiv_selection_fn 
= log(P(I_i | z_i, params))

Assuming P(I_i | z_i, Phi_i, params) is a normal CDF
    numerator = (gc + gm * beta) c_i + (gx - gm * alpha) x_i + gm (mu(z_i) - M0) + epsilon
    denominator = sqrt(8/pi)
    P(I_i | z_i, Phi_i, params) = cdf_n(numerator / denominator)
    
We can integrate out Phi_i analytically to get P(I_i | z_i, params):
    numerator = (gc + gm * beta) c_star + (gx - gm * alpha) x_star + gm (mu(z_i) - M0) + epsilon
    denominator = sqrt(8/pi + (gc + gm * beta)**2 Rc**2 + (gx - gm * alpha)**2 Rx**2 + (gm * sigma_res)**2)
    P(I_i | z_i, params) = cdf_n(numerator / denominator)
'''
# original snls selection pars: -1.7380184749999987, 0.07955165160000005, -2.60483735, 62.15588654
def log_latent_marginalized_indiv_selection_fn(mu_i, param, 
                                               selection_param=(gc, gx1, gmB, eps)):
    alpha, beta = param[0:2]
    rx, rc, sigma_res = param[2:5]
    cstar, xstar, mstar = param[5:8]
    gc, gx, gm, eps = selection_param
    
    coefs = np.array([gc + gm * beta, gx - gm * alpha, gm, eps])
    denominator = np.sqrt((8 / np.pi)
                          + (coefs[0] * rc)**2 
                          + (coefs[1] * rx)**2 
                          + (coefs[2] * sigma_res)**2)
    new_coefs = coefs / denominator
    position = np.array([cstar, xstar, mu_i + mstar, 1])
    argument = np.dot(new_coefs, position)
    return scipy.stats.norm.logcdf(argument)

'''
log_redshift_marginalized_indiv_selection_fn
P(I_i | params)
= int dz_i P(I_i | z_i, params) P(z_i)

where P(I_i | z_i, params) is np.exp(log_latent_marginalized_indiv_selection_fn)
and P(z_i) is the expected supernovae distribution over redshift
'''
def supernova_redshift_pdf(z):
    #return 260.89028471 * z**1.73916657 / 123.658 # found via fitting to data
    #return 0.1812769830095274 * z**1.372629338228674 
    s = (1 + z)**1.5   # from Dilday et al
    #s = scipy.stats.norm.pdf(z, loc=0.116, scale=0.01794)
    return s

def redshift_marginalization_integrand(z, param, selection_param, cosmo_param):
    mu = cosmology.muz(cosmo_param, z, z) # TODO: Should the z's be the same?
    return np.exp(log_latent_marginalized_indiv_selection_fn(mu, param, selection_param)) * supernova_redshift_pdf(z)

def log_redshift_marginalized_indiv_selection_fn(param, selection_param, cosmo_param):
    return np.log(scipy.integrate.quad(redshift_marginalization_integrand, 0, 1.2, args=(param, selection_param, cosmo_param))[0]) 

'''
Rubin's model:
L_rubin \propto P(D | params) P(I | D) / P(I | z, params)
    P(D | params) = L_vanilla
    P(I | D) = \prod P(I_i | D_i, params)
    P(I | z, params) = \prod P(I_i | z_i, params)
    where P(I_i | z_i, params) is log_latent_marginalized_indiv_selection_fn
'''
def rubin_log_correction(param, selection_param, phi, mu):
    log_numerator = [log_indiv_selection_fn(phi_i, selection_param) for phi_i in phi]
    log_denominator= [log_latent_marginalized_indiv_selection_fn(mu_i, param, selection_param) for mu_i in mu]
    return np.sum(log_numerator) - np.sum(log_denominator)

'''
Vincent's model:
L_Vincent \propto P(D | params) P(I | D) / P(I | params)
    P(D | params) = L_vanilla
    P(I | D) = \prod P(I_i | D_i, params)
    P(I | params) = \int dz P(I | z, params) P(z)
    where P(I | z, params) is the denominator of Rubin's correction
    note that when split over observations, 
        \int dz_i P(I_i | z_i, params) P(z_i) is the same for all i
        so P(I | params) = P(I_1 | params)**ndat
'''
def vincent_log_correction(param, selection_param, cosmo_param, phi, mu, ndat):
    log_numerator = [log_indiv_selection_fn(phi_i, selection_param) for phi_i in phi]
    #print('lognumerator: ', np.sum(log_numerator))
    #print('other part: ', ndat * log_redshift_marginalized_indiv_selection_fn(param, selection_param, cosmo_param))
    return np.sum(log_numerator) - ndat * log_redshift_marginalized_indiv_selection_fn(param, selection_param, cosmo_param)

# return array of log(weights) to compare
def weights(param, selection_param, cosmo_param, phi, mu, ndat):
    log_numerator = [log_indiv_selection_fn(phi_i, selection_param) for phi_i in phi]
    #print('lognumerator: ', np.sum(log_numerator))
    #print('other part: ', ndat * log_redshift_marginalized_indiv_selection_fn(param, selection_param, cosmo_param))
    return np.array(log_numerator) - log_redshift_marginalized_indiv_selection_fn(param, selection_param, cosmo_param)