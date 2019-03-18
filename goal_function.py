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

from numpy.polynomial import Chebyshev, chebyshev

from scipy.stats import iqr

from spectropyrometer_constants import C2
from generate_spectrum import wien_approximation


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
#    rse = std/Tave*100
    
    rse = (T_qua[1] - T_qua[0]) / (T_qua[1] + T_qua[0])
    

    return Tave,std,rse,T_left

'''
Function: goal_function
The function to minimize to obtain the correct temperature reading
Inputs:
    - X: the coefficients for the polynomial fit representation of the
    emissivity. The basis for the fit is the Chebyshev polynomials.
    - logR: the natural logarithm of the ratio of intensity at wavelength l1
    to intensity at wavelength l0
    - wl_v0, wl_v1: Chosen wavelengths vectors (nm)
    - l_min, l_max: minimum and maximum wavelengths for the filtered data (nm)    
'''
def goal_function(poly_coeff,logR,wl_v0,wl_v1,wl_min,wl_max):   
    # Create a polynomial representation with the proposed coefficients
    # Rescaling is done internally by providing the bounds l_min and l_max
    cheb = Chebyshev(poly_coeff,[wl_min,wl_max])
    
    # Calculate the emissivities at the corresponding wavelengths
    eps1 = chebyshev.chebval(wl_v1,cheb.coef)
    eps0 = chebyshev.chebval(wl_v0,cheb.coef)

    # Invert of temperature    
    with np.errstate(invalid='raise'):
        try:
            invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)

            # Temperature
            T = 1/invT
            T *= C2 * ( 1/wl_v1 - 1/wl_v0)

            ret = np.std(T)
        
        except:
            ret = 1e5
    
    if np.isnan(ret):
        return 1e5
    else:   
        return ret




'''
Function: goal_function
The function to minimize to obtain the correct temperature reading
Inputs:
    - X: the coefficients for the polynomial fit representation of the
    emissivity. The basis for the fit is the Chebyshev polynomials.
    - logR: the natural logarithm of the ratio of intensity at wavelength l1
    to intensity at wavelength l0
    - wl_v0, wl_v1: Chosen wavelengths vectors (nm)
    - l_min, l_max: minimum and maximum wavelengths for the filtered data (nm)    
'''
def mixed_goal_function(poly_coeff,logR,
                        wl_v0,wl_v1,
                        wl_min,wl_max,
                        wl_sub_vec,filtered_data,
                        bb_eps):   
    # Create a polynomial representation with the proposed coefficients
    # Rescaling is done internally by providing the bounds l_min and l_max
    cheb = Chebyshev(poly_coeff,[wl_min,wl_max])
    
    # Calculate the emissivities at the corresponding wavelengths
    eps1 = chebyshev.chebval(wl_v1,cheb.coef)
    eps0 = chebyshev.chebval(wl_v0,cheb.coef)

    # Find temperature and reconstruct base curve  
    with np.errstate(invalid='raise'):
        try:
            invT = logR - 5 * np.log(wl_v1/wl_v0) - np.log(eps0/eps1)

            # Temperature
            T = 1/invT
            T *= C2 * ( 1/wl_v1 - 1/wl_v0)

#            Tstd = np.std(T)
#            Tave = np.mean(T)
#            Trse = Tstd/Tave

            Tave,Tstd,Trse,_ = tukey_fence(T)

        
            # Calculate base curve
            Ipred = wien_approximation(wl_sub_vec,Tave,bb_eps)
            eps_vec = chebyshev.chebval(wl_sub_vec,cheb.coef)
            
            Ipred *= eps_vec
            Ipred = np.log10(np.abs(Ipred))
                        
            # Residual sum of squares and total sum of squares
            rss = np.sum((filtered_data - Ipred)**2)        
            tss = np.sum((filtered_data - np.mean(filtered_data))**2)  
            rsquared = 1 - rss/tss
            
            print(rsquared,Trse)
            
            # If rsquared is negative or zero, we have a very bad model. Penalize!
            if rsquared < 0:
                rsquared = 1e3 * np.abs(rsquared)       
            elif rsquared == 0:
                rsquared = 1e5
                
            if Trse < 0:
                Trse = 1e3 * np.abs(Trse)
                
            sqrt_term = np.abs(1-rsquared) + Trse
#            sqrt_term = Trse
            
            # Mix goal: 
            # - Rsquared should be close to one
            # - residual square error on temperature should be close to zero
#            ret = np.sqrt(sqrt_term)
            ret = sqrt_term
            
            
        except:
            ret = 1e5
    
    if np.isnan(ret):
        return 1e5
    else:   
        return ret