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

from np.polynomial import Chebyshev, chebyshev

from scipy.interpolate import splev
from scipy.stats import iqr
from scipy.optimize import minimize

from spectropyrometer_constants import C2
from spectropyrometer_constants import pix_slice,max_poly_order,rse_threshold

from goal_function import goal_function

'''
Function: tukey_fence
Descritpion: Removes outliers using Tukey fencing
Inputs:
    - Tvec: some vector
Outputs:
    - Average of vector w/o outliers
    - Standard deviation of vector w/o outliers
    - Standard error of vector w/o outliers (%)
    - Vector w/o outliers
'''      
def tukey_fence(Tvec):
    ### Exclude data w/ Tukey fencing
    T_iqr = iqr(Tvec)
    T_qua = np.percentile(Tvec,[25,75])
    
    min_T = T_qua[0] - 1.25*T_iqr
    max_T = T_qua[1] + 1.25*T_iqr
    
    T_left = Tvec[(Tvec>min_T) & (Tvec<max_T)]
    
    ### Calculate standard deviation, average, standard error
    std = np.std(T_left)
    Tave = np.mean(T_left)
    rse = std/Tave*100

    return Tave,std,rse,T_left

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
def ce_temperature(data_spl,wl_v0,wl_v1):
    Tout = []
    logR_array = []
    
    ### Standard computation
    # For each pair of wavelengths (wl0,wl1), calculate constant emissivity
    # temperature 
    for wl0,wl1 in zip(wl_v0,wl_v1):                
        # Corresponding data from the filtered data
        res0 = 10**splev(wl0,data_spl)
        res1 = 10**splev(wl1,data_spl)
      
        # Ratio of intensities
        R = res0/res1
        
        # Handle edge cases
        # Try/catch to make sure the log spits out correct values
        try:
            Ttarget = C2 * ( 1/wl1 - 1/wl0) / (np.log(R)-5*np.log(wl1/wl0))
        except:
            continue
        
        # Skip if negative or NaN
        if Ttarget < 0 or np.isnan(Ttarget):
            continue
        
        # Build vector
        Tout.append(Ttarget)
        logR_array.append(np.log(R))
    
    ### Convert to numpy arrays
    Tout = np.array(Tout)      
    logR_array = np.array(logR_array)    

    ### Returns
    Tave,std,rse,_ = tukey_fence(Tout) 
    
    return Tave,std,rse,logR_array


'''
Function: compute_temperature
Calculates the temperature
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
def compute_temperature(data_spl,cmb_pix,bins,wl_vec):
    refined_fit = False
    
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
    
    ### Calculate the temperature with the simple model
    Tave,std,rse,logR = ce_temperature(data_spl,wl_v0,wl_v1)
    print("Simple temperature model:",Tave,std,rse) 
    
    ### Do we have a "good enough" fit?   
    # If not, we assume first a linear function of emissivity and iterate
    # from there
    nunk = 2
    while rse > rse_threshold and nunk < max_poly_order:
        refined_fit = True
       
        # Define the goal function
        f = lambda pc: goal_function(pc,logR,wl_v0,wl_v1,wl_min,wl_max)
        
        # Initial values of coefficients
        pc0 = np.zeros(nunk)
        pc0[0] = 0.1
        
        # Minimization
        min_options = {'xatol':1e-10,'fatol':1e-10,'maxfev':5000}
        sol = minimize(f,pc0,method='Nelder-Mead',options = min_options)
    
        # Calculate temperature from solution
        Tave,std,rse = nce_temperature(sol.x,logR,
                    wl_v0,wl_v1,
                    wl_binm,wl_binM,
                    wl_min,
                    wl_max)
        
        print("Advanced temperature model:",Tave,std,rse,sol.x)
        
        nunk = nunk + 1
    
    return Tave,std,rse,refined_fit
    
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
def nce_temperature(poly_coeff,logR,
                    wl_v0,wl_v1,
                    wl_binm,wl_binM,
                    wl_min,
                    wl_max):  
    
    ### Polynomial representation of the emissivity
    poly_eps = Chebyshev(poly_coeff,[wl_min,wl_max])
    
    ### Emissivities at the wavelengths of interest
    eps1 = chebyshev.chebval(wl_v1,poly_eps.coef)
    eps0 = chebyshev.chebval(wl_v0,poly_eps.coef)
    
    ### Inverse temperature
    invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
    
    ### Temperature
    Tout = 1/invT
    Tout *= C2 * ( 1/wl_v1 - 1/wl_v0)

    ### Returns
    Tave,std,rse,_ = tukey_fence(Tout)
    
    print("Simple temperature model:",Tave,std,rse)  
    
    return Tave,std,rse    