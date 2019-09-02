""" 
Script for obtaining logistic regression coefficients to define selection function for SNe
"""
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import scipy

from helper_functions import cleanup



def fitSelectionFunciton(FITRES_path, SIMGEN_path):
    """
    FITRES_path: location of selected SNIa .FITRES file
    SIMGEN_path: location of bulk simulation .DAT file
    """

    # make data pandas-compatible
    cleanup.fitresCleanup(FITRES_path)
    cleanup.simgenFileCleanup(SIMGEN_path)

    FITRES_path = FITRES_path[:-7] + "_selected" + '.FITRES'
    SIMGEN_path = SIMGEN_path[:-4] + "_bulk" + '.DAT'

    # load data from smearC11 (glare type) FITRES files
    data_sel = pd.read_csv(FITRES_path, sep='\s+')  # selected SNe
    data_all = pd.read_csv(SIMGEN_path, usecols=['CID', "S2x0", 'S2x1', 'S2c', 'GENZ', 'NON1A_INDEX'], sep='\s+') # bulk simulated SNe

    # make sure all values are floats
    for name in ['S2x0', 'S2x1', 'S2c', "GENZ"]:
        data_all.loc[:, name] = data_all.loc[:,name].astype(float)    
    for name in ['zHEL', 'SIM_c', 'SIM_mB', 'SIM_x1']:
        data_sel.loc[:, name] = data_sel.loc[:,name].astype(float)

    # convert simulated x0 to mB
    S2mB = [(-2.5 * np.log10(x0) + 10.6) for x0 in data_all['S2x0']]
    data_all.loc[:,'S2mB'] = pd.Series(data=S2mB, index=data_all.index)

    # remove nan values to be able to work with the data
    data_all.replace([np.inf, -np.inf], np.nan)
    data_all = data_all.dropna(axis=0,how='any')

    intersect = np.intersect1d((data_all['CID']).astype(int), (data_sel['CID']))
    # using inbuilt Python object "in" structure setup, we can quickly compare our ID tags from data_sel and data_all:
    selectIDs = []
    for name in data_all['CID'].astype(int):
        if name in intersect:
            selectIDs.append(1) 
        else:
            selectIDs.append(0)
    len(selectIDs)

    # add to Pandas table data_all:
    data_all.loc[:,'selection_tag'] = pd.Series(data=selectIDs, index=data_all.index)
    # convert redshift values from string to floats in data_all
    data_all.loc[:,'GENZ'] = data_all['GENZ'].astype(float)

    # all non-Ia SNe are labeled with a string ID > 0. Let's create a boolean mask to remove them
    snia_mask = np.invert((data_all['NON1A_INDEX'].values.astype(int) > 0).astype(bool))
    # rewrite bulk dataset for just snia
    data_all = data_all.iloc[snia_mask]

    # remove -9 placeholder valued SNe in selection file
    snia_sel = data_sel['SIM_c'].values.astype(int) > -9
    snia_sel &= data_sel['SIM_mB'].values.astype(int) > 0
    snia_sel &= data_sel['SIM_x1'].values.astype(int) > -9

    # re-write the selected dataframe for viable SNIa:
    data_sel = data_sel.iloc[snia_sel]

    # LOGISTIC REGRESSION #
    # The sklearn modules we're going to need
    #import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.model_selection import cross_val_score

    from patsy import dmatrices  # neat helper module from patsy

    # get out the data we're interested in
    Z, phi = dmatrices('selection_tag ~ S2mB + S2c + S2x1', data_all, return_type = 'dataframe')

    # define model, and set parameter regularization to a minimum --> WHY??
    log_model = LogisticRegression(fit_intercept = False, C = 1e9)

    Z = np.ravel(Z) # flatten Z to 1D
    mdl = log_model.fit(phi, Z)


    # Next, split the observed SNe into two groups:
    # 25% data_train, 75% data_test
    phi_train, phi_test, Z_train, Z_test = train_test_split(phi, Z, train_size=0.25, test_size=0.75, random_state = 0)

    # use limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm with no parameter regularization
    learn_model = LogisticRegression(fit_intercept = False, C = 1e9, solver='lbfgs') 
    learn_model.fit(phi_train, Z_train)
    
    # store output in an accessible csv file
    coeffs = pd.DataFrame(learn_model.coef_, columns= (phi.columns))
    coeffs.to_csv('logreg_coeffs.dat', sep=',')