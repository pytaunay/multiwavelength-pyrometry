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
import spectropyrometer_constants as sc

from numpy.polynomial import Polynomial, polynomial
from statistics import tukey_fence

def goal_function(poly_coeff, logR, wl_v0, wl_v1, wl_min, wl_max): 
    '''
    Function: goal_function
    The function to minimize to obtain the correct temperature reading
    Inputs:
        - X: the coefficients for the polynomial fit representation of the
        emissivity. 
        - logR: the natural logarithm of the ratio of intensity at wavelength l1
        to intensity at wavelength l0
        - wl_v0, wl_v1: Chosen wavelengths vectors (nm)
        - l_min, l_max: minimum and maximum wavelengths for the filtered data (nm)    
    '''    
    # Create a polynomial representation with the proposed coefficients
    # Rescaling is done internally by providing the bounds l_min and l_max
    domain = np.array([wl_min,wl_max])
    pol =  Polynomial(poly_coeff,domain)
    
    # Calculate the emissivities at the corresponding wavelengths
    eps1 = polynomial.polyval(wl_v1,pol.coef)
    eps0 = polynomial.polyval(wl_v0,pol.coef)

    # Invert of temperature    
    with np.errstate(invalid='raise'):
        try:
            invT = logR - 5*np.log(wl_v1/wl_v0) - np.log(eps0/eps1)

            # Temperature
            T = 1/invT
            T *= sc.C2 * (1/wl_v1 - 1/wl_v0)

            # Calculate the coefficient of variation
            Tave, Tstd, Tmetric, _ = tukey_fence(T)
            
            ret = Tmetric
            
        except:
            ret = 1e5
    
    if np.isnan(ret):
        return 1e5
    else:   
        return ret
