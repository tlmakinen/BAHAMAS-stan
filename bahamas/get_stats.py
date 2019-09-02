import os, sys
import numpy as np
import pandas as pd
""""Takes in pandas dataframe for selected SNIa file. Returns diagonal covariance matrix for
N supernovae"""

# Define quick function for removing non-invertible covariance matrices
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def get_cov_cube(data):
    # get covariance values for matrix
    # data_cov = (data[['x1ERR', 'cERR', 'mBERR', 'COV_x1_c', 'COV_x1_x0', 'COV_c_x0', 'x0']])

        # build our covariance cube, using x0 values to get mB values
    sf = -2.5 / (np.array(data['x0']) * np.log(10))

    # pull out covariance matrix variables. Store in 3x3 blocks for each SN_i so that the matrix is invertible
    sn_cube = []
    datalist = []
    for i in range(len(sf)):
        # get covariance in terms of mB by taking derivative wrt x0
        COV_x1_mB = (data['COV_x1_x0'].values)[i] * sf[i]
        COV_c_mB =  (data['COV_c_x0'].values)[i] * sf[i]
    
        covmatrix = ([((data['cERR'].values[i])**2, data['COV_x1_c'].values[i], COV_c_mB),
                      (data['COV_x1_c'].values[i], data['x1ERR'].values[i]**2, COV_x1_mB),
                      (COV_c_mB, COV_x1_mB, (data['mBERR'].values[i])**2)])
    
        if is_pos_def(covmatrix):
            sn_cube.append(covmatrix)
            # consolidate the data for inference, keeping only invertible SNe
            datalist.append(np.array(data[['zCMB', 'zHD', 'c', 'x1', 'mB']])[i])
        
    data = np.array(datalist)  # now the data is the same length as the covariance cube diagonal

    # number of SNe with invertible covariances (only small fraction don't)
    num_sn = len(sn_cube)
    # set how big each SN's covariance block will be
    blocksize = len(sn_cube[1])
    # set up empty cov cube
    side_len = num_sn*blocksize
    bigcube = np.zeros((side_len, side_len))
    # fill cov cube diagonals with each SN covariance block
    for i in range(num_sn):    
        bigcube[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = np.array(sn_cube[i])

    sigmaC = np.matrix(bigcube)
    # put in terms of Evan's code
    return sigmaC,data
