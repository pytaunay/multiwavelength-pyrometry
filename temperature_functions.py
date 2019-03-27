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
from numpy.polynomial import Polynomial, polynomial
import spectropyrometer_constants as sc

from scipy.interpolate import splev

from scipy.optimize import minimize

from scipy.stats import normaltest


from goal_function import goal_function

from statistics import tukey_fence

def calculate_logR(data_spl, wl_v0, wl_v1):
    logR_array = []
    for wl0, wl1 in zip(wl_v0, wl_v1):                
        # Corresponding data from the filtered data
        res0 = 10**splev(wl0, data_spl)
        res1 = 10**splev(wl1, data_spl)
      
        # Ratio of intensities
        R = res0/res1
        logR = np.log(R)
        logR_array.append(logR)
        
    logR_array = np.array(logR_array)
    return logR_array

def ce_temperature(logR, wl_v0, wl_v1):
    '''
    Function: ce_temperature
    Calculates the temperature based on the averaging of multiple 
    two-wavelength predictions and Constant Emissivity (CE)
    Inputs:
        - logR The logarithm of the intensity ratio I_0 / I_1, computed 
        upstream
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
    
    ### Standard computation
    # For each pair of wavelengths (wl0,wl1), calculate constant emissivity
    # temperature 
    try:
        invT = logR - 5 *np.log(wl_v1/wl_v0)
        
        ### Temperature
        Tout = 1/invT
        Tout *= sc.C2 * ( 1/wl_v1 - 1/wl_v0)
    
        ### Returns
        Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout, method = 'dispersion')
        print('Coeffs: ', '[]', '\t p-value:',normaltest(Tleft)[1])
        
    except:
        Tave,std,rse = 1e5 * np.ones(3)
    
    return Tave, Tstd, Tmetric   
   
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
    # Create a polynomial representation with the proposed coefficients
    # Rescaling is done internally by providing the bounds l_min and l_max
    domain = np.array([wl_min,wl_max])
    pol =  Polynomial(poly_coeff,domain)
    
    # Calculate the emissivities at the corresponding wavelengths
    eps1 = polynomial.polyval(wl_v1,pol.coef)
    eps0 = polynomial.polyval(wl_v0,pol.coef)
    
    ### Inverse temperature
    try:
        invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
        
        ### Temperature
        Tout = 1/invT
        Tout *= sc.C2 * ( 1/wl_v1 - 1/wl_v0)
    
        ### Returns
        Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout, method = 'dispersion')
        print('Coeffs: ', poly_coeff, '\t p-value:',normaltest(Tleft)[1])
    except:
        Tave, Tstd, Tmetric = 1e5 * np.ones(3)
    
    return Tave, Tstd, Tmetric      



def optimum_temperature(data_spl, cmb_pix, pix_vec, wl_vec, order):    
    '''
    Function: optimum_temperature
    Calculates the temperature based on the assumption of a polynomial order
    Inputs:
        - data_spl Spline representation of the filtered intensity data
        - cmb_pix Pixels chosen for each pixel bin
        - pix_vec Overall pixel vector
        - wl_vec Vector of wavelengths (nm)
    Ouputs:
        - Predicted temperature from averaging (K)
        - Standard deviation (K)
        - Standard deviation (%)
        - Flag indicating if advanced method was used
    '''
    bins = pix_vec[0::sc.pix_slice]
    wl_sub_vec = wl_vec[pix_vec]
    
    # Minimum and maximum wavelengths
    wl_min = np.min(wl_sub_vec)
    wl_max = np.max(wl_sub_vec)

    # Which wavelengths are associated with the pixel combinations?
    wl_v0 = wl_vec[cmb_pix[:,0]]
    wl_v1 = wl_vec[cmb_pix[:,1]] 

    # Create the [lambda_min,lambda_max] pairs that delimit a "bin"
    wl_binm = wl_vec[bins]
    wl_binM = wl_vec[bins[1::]]
    wl_binM = np.append(wl_binM,wl_vec[-1])
    
    ### Calculate intensity ratio
    logR = calculate_logR(data_spl, wl_v0, wl_v1)
    
    ### Which order are we using?
    if order == 0:
        # If emissivity is constant, calculate the temperature with the simple model
        sol = None
        Tave, Tstd, Tmetric = ce_temperature(logR,wl_v0,wl_v1)
    else:    
        # Otherwise, optimization routine on the coefficients of epsilon
        # Define the goal function
        f = lambda pc: goal_function(pc, logR, wl_v0, wl_v1, wl_min, wl_max)
        
        # Initial values of coefficients
        pc0 = np.zeros(order+1)
        pc0[0] = sc.eps0
    
        # Minimization
        min_options = {'xatol':1e-15, 'fatol':1e-15, 'maxfev':5000} # Nelder-Mead
        sol = minimize(f, pc0, method = 'Nelder-Mead', options = min_options)
    
        # Calculate temperature from solution
        Tave, Tstd, Tmetric = nce_temperature(sol.x,logR,
                            wl_v0,wl_v1,
                            wl_binm,wl_binM,
                            wl_min,
                            wl_max)
    return Tave, Tstd, Tmetric, sol 
      




