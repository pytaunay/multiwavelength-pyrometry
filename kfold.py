# Copyright (C) 2019 Pierre-Yves Taunay
# 
# This program is free software: you can redistribute it andor modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https:/www.gnu.org/licenses/>.
# 
# Contact info: https:/github.com/pytaunay
# 
# Source: https:/github.com/pytaunay/ILX526A

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

import warnings
import spectropyrometer_constants as sc
import temperature_functions as tf

from pixel_operations import choose_pixels, generate_combinations

from goal_function import goal_function

def training(data_spl, pix_sub_vec, train_idx, wl_vec):
    '''
    Training phase: optimize each emissivity model individually on a training 
    subset.
    Inputs:
        - data_spl: spline representation of the filtered intensity
        - pix_sub_vec: the pixel indices that are used to define the filtered 
        data
        - train_idx: the array indices of pix_sub_vec that are used for 
        training 
        - wl_vec: the full wavelength vector
    '''
    ### Get the pixels and wavelengths we will use for training
    train_pix = pix_sub_vec[train_idx]
    wl_sub_vec = wl_vec[pix_sub_vec]
    
    ### Generate pairs of pixels
    chosen_pix = choose_pixels(train_pix, bin_method='median')
    cmb_pix = generate_combinations(chosen_pix, pix_sub_vec)

    ### Pixel operations
    bins = pix_sub_vec[0::sc.pix_slice]
    
    # Minimum and maximum wavelengths
    wl_min = np.min(wl_sub_vec)
    wl_max = np.max(wl_sub_vec)

    # Which wavelengths are associated with the pixel combinations?
    wl_v0 = wl_vec[cmb_pix[:,0]]
    wl_v1 = wl_vec[cmb_pix[:,1]] 

    # Create the [lambda_min,lambda_max] pairs that delimit a "bin"
    wl_binm = wl_vec[bins]
    wl_binM = wl_vec[bins[1::]]
    wl_binM = np.append(wl_binM, wl_vec[-1])
    

    ### Test multiple models of emissivity until we satisfy the threshold for 
    ### the interquartile dispersion
    logR = tf.calculate_logR(data_spl, wl_v0, wl_v1)
    
    # 1. Calculate the temperature with the simple model
    Tave, Tstd, Tmetric = tf.ce_temperature(logR, wl_v0, wl_v1)
    print("Simple temperature model:", Tave, Tstd, Tmetric) 
    
    # 2. Calculate the temperature with a variable emissivity 
    # Do we have a "good enough" fit?   
    # If not, we assume first a linear function of emissivity and iterate
    # from there
    sol = None
    nunk = 1 
    
    model_training = []
    
    while Tmetric > sc.threshold and nunk < sc.max_poly_order:
        # Define the goal function
        f = lambda pc: goal_function(pc, logR, wl_v0, wl_v1, wl_min, wl_max)
        
        # Initial values of coefficients
        pc0 = np.zeros(nunk+1)
        pc0[0] = sc.eps0  
        
        # Minimization of the coefficient of variation: Nelder-Mead
        min_options = {'xatol':1e-15, 'fatol':1e-15, 'maxfev':5000} 
        sol = minimize(f, pc0,
                       method='Nelder-Mead',
                       options=min_options)

        # Calculate temperature from solution
        Tave, Tstd, Tmetric = tf.nce_temperature(sol.x, logR,
                    wl_v0, wl_v1,
                    wl_binm, wl_binM,
                    wl_min,
                    wl_max)
        
        print("Advanced temperature model:", Tave, Tstd, Tmetric, sol.x)
        
        nunk = nunk + 1
        model_training.append(sol.x)
    
    return model_training

def testing(data_spl, pix_sub_vec, test_idx, wl_vec, model_training):
    '''
    Tests all models on the test pixels.
    Inputs:
        - data_spl: spline representation of the filtered intensity
        - pix_sub_vec: the pixel indices that are used to define the filtered 
        data
        - train_idx: the array indices of pix_sub_vec that are used for 
        training 
        - wl_vec: the full wavelength vector
        - model_training: the coefficients for the different models proposed
    '''
    ### This array will contain the resulting metric for each model, for a 
    ### single ensemble of test pixels
    model_metric = []
    
    ### Get the pixels we will use for testing
    test_pix = pix_sub_vec[test_idx]
    wl_sub_vec = wl_vec[pix_sub_vec]
    ### Generate pairs of pixels
    chosen_pix = choose_pixels(test_pix, bin_method='median')
    cmb_pix = generate_combinations(chosen_pix,pix_sub_vec)

    ### Pixel operations
    bins = test_pix[0::sc.pix_slice]
    
    # Minimum and maximum wavelengths
    wl_min = np.min(wl_sub_vec)
    wl_max = np.max(wl_sub_vec)

    # Which wavelengths are associated with the pixel combinations?
    wl_v0 = wl_vec[cmb_pix[:,0]]
    wl_v1 = wl_vec[cmb_pix[:,1]] 

    # Create the [lambda_min,lambda_max] pairs that delimit a "bin"
    wl_binm = wl_vec[bins]
    wl_binM = wl_vec[bins[1::]]
    wl_binM = np.append(wl_binM, wl_vec[-1])
    
    ### Apply tests
    logR = tf.calculate_logR(data_spl, wl_v0, wl_v1)
    
    # 1. Constant emissivity
    Tave, Tstd, Tmetric = tf.ce_temperature(logR, wl_v0, wl_v1)
    
    model_metric.append(Tmetric)
    
    # 2. All other orders
    for lcoeff in model_training:
        Tave, Tstd, Tmetric = tf.nce_temperature(lcoeff, logR, wl_v0, wl_v1,
                                              wl_binm, wl_binM,
                                              wl_min, wl_max)
        
        model_metric.append(Tmetric)

    return model_metric

def order_selection(data_spl,
                       pix_sub_vec,wl_vec,
                       bb_eps):
    '''
    Select the correct polynomial order by performing the k-fold cross-valida-
    tion method. 
    The k-folds are generated randomly based on a randomly-generated seed.
    They split the indices for the wavelengths into training and testing data-
    sets. For example, the following indices could be used for a vector of 
    indices of size 2900 that starts from 50:
        original: [50, 51, 52, ... , 2949]
        training: [50, 51, 52, ..., 2800]
        testing: [2801, ..., 2949]: 
    Inputs:
        - data_spl: spline representation of the filtered data
        - pix_sub_vec: indices of wavelengths over which the spline is defined
        - wl_vec: all input wavelengths
        - bb_eps: a black-body emissivity function
    '''
    ### Generate a training and testing dataset for the pixels themselves
    n_splits = sc.ksplits
    kf = KFold(n_splits = n_splits, shuffle=True, 
               random_state = np.random.randint(0,1000))
    metric_array = np.zeros((n_splits, sc.max_poly_order+1))
    metric_all = []

    ### For all pairs of training and testing datasets...
    for train_idx, test_idx in kf.split(pix_sub_vec):     
        print("-------TRAINING--------")
        ### Training
        model_training = training(data_spl, pix_sub_vec, train_idx, wl_vec)
                
        print("-------TESTING--------")
        ### Testing
        model_metric = testing(data_spl, pix_sub_vec, test_idx, wl_vec, 
                               model_training)
        
        metric_all.append(model_metric)
        
    for idx in range(len(metric_all)):
        nelem = len(metric_all[idx])
        print(metric_all[idx])
        metric_array[idx,0:nelem] = np.array(metric_all[idx])

    # Count number of entries that are non-zero for each polynomial order
    for col in metric_array.T:
        nz = np.count_nonzero(col)
        
        # If we do not have a column full of numbers, it means that the
        # proposed emissivity polynomial yields a minimum in data dispersion 
        # at a lower polynomial order. We should favor that lower polynomial
        # order
        if nz < n_splits:
            col[col == 0] = 1e1

    # Ignore zeros for the mean
    metric_array[metric_array == 0] = np.nan

    # Suppress "mean of empty slice" warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        mean = np.nanmean(metric_array, axis=0)
       
    
        
    poly_order = np.nanargmin(mean)

    print("Mean of all k-folds:", mean)
    print("Chosen polynomial order: ", poly_order)
 
    return poly_order