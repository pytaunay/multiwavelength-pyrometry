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

from scipy.interpolate import splev

from scipy.optimize import minimize, lsq_linear, basinhopping

from spectropyrometer_constants import C2
from spectropyrometer_constants import pix_slice,max_poly_order,rse_threshold

from goal_function import goal_function, mixed_goal_function

from statistics import tukey_fence

def ce_temperature(data_spl,wl_v0,wl_v1):
    '''
    Function: ce_temperature
    Calculates the temperature based on the averaging of multiple two-temperature
    predictions and Constant Emissivity (CE)
    Inputs:
        - data_spl Spline representation of the filtered intensity data
        - wl_v0, wl_v1 Vector of wavelengths chosen
    Ouputs:
        - Predicted temperature from averaging (K)
        - Standard deviation (K)
        - Standard deviation (%)
        - Natural logarithm of ratio of intensities of two wavelengths. Useful for
        the non-constant emissivity case as well and avoids having to recalculate
        it.
    '''    
    Tout = []
    logR_array = []
    
    ### Standard computation
    # For each pair of wavelengths (wl0,wl1), calculate constant emissivity
    # temperature 
    for wl0, wl1 in zip(wl_v0, wl_v1):                
        # Corresponding data from the filtered data
        res0 = 10**splev(wl0, data_spl)
        res1 = 10**splev(wl1, data_spl)
      
        # Ratio of intensities
        R = res0/res1
        logR = np.log(R)
        
        # Handle edge cases
        # Try/catch to make sure the log spits out correct values
        try:
            Ttarget = C2 * ( 1/wl1 - 1/wl0) / (logR-5*np.log(wl1/wl0))
        except:
            continue
        
        # Skip if negative or NaN
        if Ttarget < 0 or np.isnan(Ttarget):
            continue
        
        # Build vector
        Tout.append(Ttarget)
        logR_array.append(logR)
    
    ### Convert to numpy arrays
    Tout = np.array(Tout)      
    logR_array = np.array(logR_array)    

    ### Returns
    Tave, Tstd, Tmetric, _ = tukey_fence(Tout) 
    
    return Tave, Tstd, Tmetric, logR_array
   

def compute_poly_temperature(data_spl,cmb_pix,pix_vec,wl_vec,order):    
    return 0
#    '''
#    Function: compute_poly_temperature
#    Calculates the temperature based on the assumption of a polynomial order
#    Inputs:
#        - data_spl Spline representation of the filtered intensity data
#        - cmb_pix Pixels chosen for each pixel bin
#        - pix_vec Overall pixel vector
#        - wl_vec Vector of wavelengths (nm)
#    Ouputs:
#        - Predicted temperature from averaging (K)
#        - Standard deviation (K)
#        - Standard deviation (%)
#        - Flag indicating if advanced method was used
#    '''
#    bins = pix_vec[0::pix_slice]
#    wl_sub_vec = wl_vec[pix_vec]
#    
#    # Minimum and maximum wavelengths
#    wl_min = np.min(wl_sub_vec)
#    wl_max = np.max(wl_sub_vec)
#    wl_ave = np.average(wl_vec)
#
#    # Which wavelengths are associated with the pixel combinations?
#    wl_v0 = wl_vec[cmb_pix[:,0]]
#    wl_v1 = wl_vec[cmb_pix[:,1]] 
#
#
#    # Create the [lambda_min,lambda_max] pairs that delimit a "bin"
#    wl_binm = wl_vec[bins]
#    wl_binM = wl_vec[bins[1::]]
#    wl_binM = np.append(wl_binM,wl_vec[-1])
#    
#    ### Calculate the temperature with the simple model
#    Tave,std,rse,logR = ce_temperature(data_spl,wl_v0,wl_v1)
#    sol = None
#    
#    # Define the goal function
#    filtered_data = splev(wl_sub_vec,data_spl)
#    bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
#    f = lambda pc: mixed_goal_function(pc,logR,
#                                       wl_v0,wl_v1,
#                                       wl_min,wl_max,
#                                       wl_sub_vec,filtered_data,
#                                       bb_eps)
##    f = lambda pc: goal_function(pc,logR,wl_v0,wl_v1,wl_min,wl_max)
#    
#    # Initial values of coefficients
#    pc0 = np.zeros(order+1)
#    pc0[0] =  1     
#
#    # Minimization
#    min_options = {'xatol':1e-15,'fatol':1e-15,'maxfev':5000} # Nelder-Mead
#    sol = minimize(f,pc0,method='Nelder-Mead',options=min_options)
##    def fconst(coeffs,wl_sub_vec,wl_min,wl_max):
##        cheb = Chebyshev(coeffs,[wl_min,wl_max])
##        val = chebyshev.chebval(wl_sub_vec,cheb.coef)       
##        sgn = np.sign(val)
##        sgn = np.prod(sgn)
##        return sgn
#    
##    cons = ({'type': 'ineq', 'fun': lambda coeff: fconst(coeff,wl_sub_vec,wl_min,wl_max) })
##    min_options = {'tol':1e-15,'maxiter':5000,'rhobeg':100} # COBYLA
##    sol = minimize(f,pc0,method='COBYLA',constraints=cons,options=min_options)
#    
##    sol = basinhopping(f,pc0)
#    
#    # Calculate temperature from solution
#    Tave,std,rse = nce_temperature(sol.x,logR,
#                        wl_v0,wl_v1,
#                        wl_binm,wl_binM,
#                        wl_min,
#                        wl_max)
#    
#    return Tave,std,rse,sol    



def nce_temperature(poly_coeff,logR,
                    wl_v0,wl_v1,
                    wl_binm,wl_binM,
                    wl_min,
                    wl_max):  
    '''
    Function: nce_temperature
    Calculates the temperature based on a Non-Constant Emissivity (NCE).
    The emissivity is modeled with a Chebyshev polynomial of order N where N 
    is determined by a separate routine
    Inputs:
        - 
    Outputs:
        - Predicted temperature from averaging (K)
        - Standard deviation (K)
        - Standard deviation (%)
     '''   
    ### Polynomial representation of the emissivity
    poly_eps = Chebyshev(poly_coeff,[wl_min,wl_max])
    
    ### Emissivities at the wavelengths of interest
    eps1 = chebyshev.chebval(wl_v1,poly_eps.coef)
    eps0 = chebyshev.chebval(wl_v0,poly_eps.coef)
    
    ### Inverse temperature
    try:
        invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
        
        ### Temperature
        Tout = 1/invT
        Tout *= C2 * ( 1/wl_v1 - 1/wl_v0)
    
        ### Returns
        Tave,std,rse,_ = tukey_fence(Tout)
    except:
        Tave,std,rse = 1e5 * np.ones(3)
    
    return Tave,std,rse    