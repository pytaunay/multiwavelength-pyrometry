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
from numpy.polynomial import Chebyshev,chebyshev

import warnings

from sklearn.model_selection import KFold

from pixel_operations import choose_pixels, generate_combinations
from temperature_functions import ce_temperature, nce_temperature
from generate_spectrum import wien_approximation
from spectropyrometer_constants import pix_slice, max_poly_order, cv_threshold

def training(data_spl, pix_sub_vec, train_idx, wl_vec):
    '''
    Inputs:
        - data_spl: spline representation of the filtered intensity
        - pix_sub_vec: the pixel indices that are used to define the filtered 
        data
        - train_idx: the array indices of pix_sub_vec that are used for 
        training 
        - wl_vec: the full wavelength vector
    '''
    ### Get the pixels we will use for training
    train_pix = pix_sub_vec[train_idx]
    ### Generate pairs of pixels
    chosen_pix = choose_pixels(train_pix,bin_method='average')
    cmb_pix = generate_combinations(chosen_pix,pix_sub_vec)

    ### Pixel operations
    wl_sub_vec = wl_vec[pix_sub_vec]
    bins = pix_sub_vec[0::pix_slice]
    
    # Minimum and maximum wavelengths
    wl_min = np.min(wl_vec)
    wl_max = np.max(wl_vec)

    # Which wavelengths are associated with the pixel combinations?
    wl_v0 = wl_vec[cmb_pix[:,0]]
    wl_v1 = wl_vec[cmb_pix[:,1]] 

    # Create the [lambda_min,lambda_max] pairs that delimit a "bin"
    wl_binm = wl_vec[bins]
    wl_binM = wl_vec[bins[1::]]
    wl_binM = np.append(wl_binM,wl_vec[-1])
    

    ### Test multiple models of emissivity until we satisfy the threshold for 
    ### the coefficient of variation
#    Tave,std,rse,refined_fit,sol,sol_all = compute_temperature(data_spl,
#                                                                   cmb_pix,
#                                                                   pix_sub_vec,
#                                                                   wl_vec)

    # 1. Calculate the temperature with the simple model
    Tave,std,rse,logR = ce_temperature(data_spl,wl_v0,wl_v1)
    print("Simple temperature model:",Tave,std,rse) 
    sol = None
    
    ### Do we have a "good enough" fit?   
    # If not, we assume first a linear function of emissivity and iterate
    # from there
    nunk = 2
    perturb = False
    sol_all = []
    while rse > rse_threshold and nunk < max_poly_order:
        refined_fit = True

        # What is our previous standard error?
        rse_nm1 = rse
        
        # Define the goal function
        f = lambda pc: goal_function(pc,logR,wl_v0,wl_v1,wl_min,wl_max)
        
        # Initial values of coefficients
        pc0 = np.zeros(nunk)
        pc0[0] =  0.5     
        
        if perturb:
            pc0[0] = 0
            while pc0[0] == 0:
                pc0[0] = np.random.sample()
               

        # Minimization
        min_options = {'xatol':1e-15,'fatol':1e-15,'maxfev':5000} # Nelder-Mead
        sol = minimize(f,pc0,method='Nelder-Mead',options=min_options)

        # Calculate temperature from solution
        Tave,std,rse = nce_temperature(sol.x,logR,
                    wl_v0,wl_v1,
                    wl_binm,wl_binM,
                    wl_min,
                    wl_max)
        
        print("Advanced temperature model:",Tave,std,rse,sol.x,pc0[0])
        
        nunk = nunk + 1
        sol_all.append(sol.x)
    
    return Tave,std,rse,refined_fit,sol,sol_all

def testing():
        ### Testing on all models
        test_pix = pix_sub_vec[test]
        chosen_pix = choose_pixels(test_pix,bin_method='average')
        cmb_pix = generate_combinations(chosen_pix,pix_sub_vec)

        bins = test_pix[0::pix_slice]
        
        # Minimum and maximum wavelengths
        wl_min = np.min(wl_sub_vec)
        wl_max = np.max(wl_sub_vec)
        wl_ave = np.average(wl_sub_vec)
    
        # Which wavelengths are associated with the pixel combinations?
        wl_v0 = wl_vec[cmb_pix[:,0]]
        wl_v1 = wl_vec[cmb_pix[:,1]] 
    
    
        # Create the [lambda_min,lambda_max] pairs that delimit a "bin"
        wl_binm = wl_vec[bins]
        wl_binM = wl_vec[bins[1::]]
        wl_binM = np.append(wl_binM,wl_vec[-1])
        
        # Order 0
        Tave_test,std,rse,logR = ce_temperature(data_spl,wl_v0,wl_v1)
        
        Ipred = wien_approximation(wl_sub_vec,Tave_test,bb_eps)
        Ipred = np.log10(Ipred)
        
        # Calculate R2
        rss = np.sum((filtered_data - Ipred)**2)        
        tss = np.sum((filtered_data - np.mean(filtered_data))**2)  
        rsquared = 1 - rss/tss  
        
        # If rsquared is negative or zero, we have a very bad model. Penalize!
        if rsquared < 0:
            rsquared = 1e3 * np.abs(rsquared)       
        elif rsquared == 0:
            rsquared = 1e5
        
        # Geometric average of Rsquared and RSE
#        mse_test = np.sqrt(np.abs(1-rsquared) * rse/100)
#        mse_test = 1/2 * (np.abs(1-rsquared) + rse/100)
        mse_test = rse
        mse_single.append(mse_test)
        
        # All other orders
        for coeff in sol_all:
            Tave_test,std_test,rse_test = nce_temperature(coeff,logR,
                    wl_v0,wl_v1,
                    wl_binm,wl_binM,
                    wl_min,
                    wl_max)
        
            Ipred = wien_approximation(wl_sub_vec,Tave_test,bb_eps)
            
            wl_min = np.min(wl_sub_vec)
            wl_max = np.max(wl_sub_vec)

            cheb = Chebyshev(coeff,[wl_min,wl_max])
            eps_vec = chebyshev.chebval(wl_sub_vec,cheb.coef)
                    
            Ipred *= eps_vec
            Ipred = np.log10(np.abs(Ipred))
            
            # Calculate R2
            rss = np.sum((filtered_data - Ipred)**2)        
            tss = np.sum((filtered_data - np.mean(filtered_data))**2)  
            rsquared = 1 - rss/tss  
            
            # If rsquared is negative or zero, we have a very bad model. Penalize!
            if rsquared < 0:
                rsquared = 1e3 * np.abs(rsquared)       
            elif rsquared == 0:
                rsquared = 1e5
            
            if rse_test < 0:
                rse_test = 1e3 * np.abs(rse_test)
            
            # Geometric average of Rsquared and RSE
#            mse_test = np.sqrt(np.abs(1-rsquared) * rse_test/100)
#            mse_test = 1/2 * (np.abs(1-rsquared) + rse_test/100)
            mse_test = rse_test/100
            
            mse_single.append(mse_test)


def order_selection(data_spl,filtered_data,
                       pix_sub_vec,wl_vec,
                       bb_eps):
    ### Get the wavelengths used as the support for the data
    wl_sub_vec = wl_vec[pix_sub_vec]

    ### Generate a training and testing dataset for the pixels themselves
    kf = KFold(n_splits=max_poly_order+1,shuffle=True)
    mse_all = []
    mse_array = np.zeros((max_poly_order+1,max_poly_order+1))

    ### For all pairs of training and testing datasets...
    for train, test in kf.split(pix_sub_vec):
        mse_single = []

        ### Training

                
        mse_all.append(mse_single)
    
        
    for idx in range(len(mse_all)):
        nelem = len(mse_all[idx])
        print(mse_all[idx])
        mse_array[idx,0:nelem] = np.array(mse_all[idx])

    # Ignore zeros for the mean
    mse_array[mse_array == 0] = np.nan

    # Suppress "mean of empty slice" warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(mse_array,axis=0)
        
    poly_order = np.nanargmin(mean)

    print("Mean of all k-folds:", mean)
    print("Chosen polynomial order: ", poly_order)
 
    return poly_order


