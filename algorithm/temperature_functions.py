# MIT License
# 
# Copyright (c) 2020 Pierre-Yves Camille Regis Taunay
#  
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from numpy.polynomial import Polynomial, polynomial
import algorithm.spectropyrometer_constants as sc

from scipy.interpolate import splev

from scipy.optimize import minimize

from algorithm.goal_function import goal_function

from algorithm.statistics import tukey_fence

def calculate_logR(data_spl, wl_v0, wl_v1):
    logR_array = []
    for wl0, wl1 in zip(wl_v0, wl_v1):                
        # Corresponding data from the filtered data
        res0 = np.exp(splev(wl0, data_spl))
        res1 = np.exp(splev(wl1, data_spl))
      
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
        Tout *= sc.C2 * (1/wl_v1 - 1/wl_v0)
    
        ### Returns
#        Tout = Tout[Tout>0]
        Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout)
    # If there is some issue with the computation, avoid this data point    
    except:
        Tave,Tstd,Tmetric = 1e5 * np.ones(3)
    
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
        Tout *= sc.C2 * (1/wl_v1 - 1/wl_v0)
    
        ### Returns
#        Tout = Tout[Tout>0]
        Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout)
#        print('Coeffs: ', poly_coeff, '\t p-value:',normaltest(Tleft)[1])
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
        min_options = {'xatol':1e-15, 'fatol':1e-15, 'maxfev':20000} # Nelder-Mead
        sol = minimize(f, pc0, method = 'Nelder-Mead', options = min_options)
    
        # Calculate temperature from solution
        Tave, Tstd, Tmetric = nce_temperature(sol.x,logR,
                            wl_v0,wl_v1,
                            wl_binm,wl_binM,
                            wl_min,
                            wl_max)
    return Tave, Tstd, Tmetric, sol 
      




