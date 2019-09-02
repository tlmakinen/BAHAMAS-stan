'''
Sim script for data generation for BAHAMAS STAN toy model
'''

import numpy as np
import pandas as pandas
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.integrate import quad

from bahamas import cosmology

# Parameters to be inferred
omegam = 0.3
omegade = 0.7
c_light = 299792.0 # Speed of light in km/s
h = 0.72 # Hubble parameter  # EDIT from 0.72
H_0 = 100.0 # Hubble constant
M_0 = -19.3
sigma_int = 0.1

# selection function parameters
m_cut = 24
sigma_cut = 0.5

# selection function
def selection_fn(m, m_cut, sigma_cut):
    argument = (m_cut - m) / sigma_cut
    return scipy.stats.norm.cdf(argument)


# cosmo functions
def distance_modulus(z, matter_density,d_energy_density):
    #print(type(z),type(matter_density),type(d_energy_density),type(c_light),type(h))
    eta = (-5.0 * np.log10(H_0*h/c_light)) + 25.0
    #print(eta)
    #print(z,matter_density,d_energy_density)
    theoretical_distance_modulus = eta + (5*np.log10(luminosity_distance(z, matter_density,d_energy_density)))

    return theoretical_distance_modulus

def luminosity_distance(z, matter_density, d_energy_density):
    '''
    An integral to be evaluated based on the redshift value and density parameters
    '''
    curvature_density = 1.0 - matter_density - d_energy_density
    #print(curvature_density, matter_density, d_energy_density)
    def luminosity_integrand(z):
        '''
        The integrand part of the luminosity_distance
        '''
        matter_term = matter_density * ((1.0 + z)**3.0)
        energy_term = d_energy_density
        curvature_term = curvature_density * ((1.0 + z)**2.0)
        integrand = (matter_term + energy_term + curvature_term)**(-0.5)
        return integrand

    integrand,err = quad(luminosity_integrand,0.0,z)


    l_distance = np.sqrt(np.abs(curvature_density)) * integrand
    #define front term within the conditions to avoid the divide by zero error.
    if(curvature_density>0):
        front_term = ((1.0 + z)/(np.sqrt(np.abs(curvature_density))))
        l_distance =  front_term * np.sinh(l_distance)
    elif(curvature_density <0):
        front_term = ((1.0 + z)/(np.sqrt(np.abs(curvature_density))))
        l_distance = front_term * np.sin(l_distance)
    elif(curvature_density == 0):
        #We don't use front_term * l_distance here due to the divide by zero error,
        #instead we use the simplified equation that appears when curvature is 0.
        l_distance = (1 + z) * integrand


    return l_distance



# Create data

# size of dataset
ndat = 500

def generate_sn1a():
    # Step 1: draw latent redshift z~(1+z)^1.5
    q = ss.powerlaw.rvs(2.5, loc=0, scale=2.3) # q = z+1 from Dilday et al
    while q-1 < 0:
        q = ss.powerlaw.rvs(2.5, loc=0, scale=2.3)
    z_i = q-1

    # Step 2: generate distance modulus to SN1a
    mu_i = distance_modulus(z_i,omegam,omegade)



    # Step 2: draw latent magnitude, m:
    m_i = np.random.normal(loc=M_0 + mu_i, scale=sigma_int)

    # Step 3: make selection cuts
    selection_prob_i = selection_fn(m_i, m_cut, sigma_cut)

    # draw random number to simulate whack chance of SN1a being missed
    rand = np.random.uniform(low=0, high=1.0)

    if selection_prob_i > rand:
        selection_tag  = 1
    else:
        selection_tag = 0

    return (z_i, m_i, selection_prob_i, selection_tag)

# Generate supernovae until 500 selected
num_observed = 0

sim_z = []
sim_m = []
sim_sel_prob = []
sim_sel_tag = []

while num_observed < ndat:
    sim = generate_sn1a()
    sim_z.append(sim[0])
    sim_m.append(sim[1])
    sim_sel_prob.append(sim[2])
    sim_sel_tag.append(sim[3])

    if sim[3] > 0:
        num_observed += 1

# convert to np arrays
sim_z = np.array(sim_z)
sim_m = np.array(sim_m)
sim_sel_prob = np.array(sim_sel_prob)
sim_sel_tag = np.array(sim_sel_tag)

# for selected SN1a
all_simulated_data = (np.array([sim_z,sim_m, sim_sel_prob, sim_sel_tag]).transpose())
mask = (all_simulated_data[:, 3] > 0).astype(bool)
miss_mask = np.invert(mask)

# make plot
# plot selection function for selected vs missed SN1a
fig = plt.figure(figsize=(8,8))
fig.suptitle('Simulated SN1a Selection Probability')
# selection fn in mB
plt.subplot(111)
plt.xlabel('$m$')
plt.ylabel('$p$(selection)')
plt.scatter(np.array(sim_m)[miss_mask], np.array(sim_sel_prob)[miss_mask], 
                    color='r', marker ='.', s=3, label='{} Missed SN1a'.format(np.sum(miss_mask)))
plt.scatter(np.array(sim_m)[mask], np.array(sim_sel_prob)[mask], 
                    color='b', marker ='.', s=3, label='{} Selected SN1a'.format(np.sum(mask)))
plt.legend(loc="lower left")

plt.savefig('simuated_selection_fn.png')

# save data

headers = "z m sel_prob sel_tag"
#all_simulated_data = (np.array([sim_z,sim_mb,sim_dmb,sim_x1,sim_dx1,sim_c,sim_dc,sim_l,sim_b,sim_true_mb,sim_true_x1,sim_true_c, sim_selection_prob, sim_selection_tag]).transpose())

# now take the slice of DUMP file where the SN1a are observed
mask = (all_simulated_data[:, 3] > 0).astype(bool)

# make dump file for all data
simulated_data = (np.array([sim_z,sim_m, sim_sel_prob, sim_sel_tag]).transpose())

# make selected file for inference
selected_simulated_data = simulated_data[mask]

# full dump file
np.savetxt('toy_dump.txt', simulated_data, delimiter=' ',header = headers,comments="")
# selected dataset
np.savetxt('toy_selected.txt',selected_simulated_data, delimiter=' ',header = headers,comments="")
