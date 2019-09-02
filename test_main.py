'''
Sim script for data generation for BAHAMAS STAN toy model
'''

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.integrate import quad

import pystan
#import seaborn as sns
#sns.set()  # Nice plot aesthetic

from bahamas import cosmology

# Parameters to be inferred
omegam = 0.3
omegade = 0.7
c_light = 299792.0 # Speed of light in km/s
h = 0.72 # Hubble parameter  # EDIT from 0.72
H_0 = 100.0 # Hubble constant
M_0 = -19.3
sigma_int = 0.1

parameters = ['M_0', 'sigma_int', 'omegam', 'omegade']
ndim = len(parameters)

# selection function parameters
m_cut = 24
sigma_cut = 0.5

# pull in data 
data_names = ['z', 'm', 'sel_prob', 'sel_tag']
data = pd.read_csv('toy_selected.txt', header=0, sep='\s+')

zHD = data['z'].values
# add second z column so that we have a zHD value
data.insert(loc=1, column='zHD', value=zHD)

z = np.array(data[['z']])
m = np.array(data['m'])

# put data in python dictionary for stan compiler
data = {'N': len(m), 'ndim': ndim, 'z': z, 'y': m}

# compute mu
#mu = cosmology.muz()

toy_model = """

functions {
    real normal_density(real x, // Function argument
                        real xc, // Complement of function argument
                                // on the domain (defined later)
                        real[] theta, // parameters
                        real[] x_r, // data (real)
                        int[] x_i) { // data (integer)
    real mu = theta[1];
    real sigma = theta[2];
    return 1 / (sqrt(2 * pi()) * sigma) * exp(-0.5 * ((x - mu) / sigma)^2);
    }

}


data {
    int N;
    real y[N];
}

transformed data {
    real x_r[0];
    int x_i[0];
}

parameters {
    real mu;
    real<lower = 0.0> sigma;
    real left_limit;
}

model {
    mu ~ normal(0, 1);
    sigma ~ normal(0, 1);
    left_limit ~ normal(0, 1);
    target += normal_lpdf(y | mu, sigma);
    target += log(integrate_1d(normal_density,
                                left_limit,
                                positive_infinity(),
                                { mu, sigma }, x_r, x_i));
}


"""

# Compile the model
sm = pystan.StanModel(file='toy.stan')

# Train the model and generate samples
fit = sm.sampling(data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101)

fit.plot()