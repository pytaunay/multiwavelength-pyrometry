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

from spectro_pyrometer_constants import C2

'''
Function: goal_function
The function to minimize to obtain the correct temperature reading
Inputs:
    - X: the coefficients for the polynomial fit representation of the
    emissivity. The basis for the fit is the Chebyshev polynomials.
    - logR: the natural logarithm of the ratio of intensity at wavelength l1
    to intensity at wavelength l0
    - l1, l0: wavelengths (nm)
    - l_min, l_max: minimum and maximum wavelengths for the filtered data (nm)    
'''
def goal_function(poly_coeffs,logR,wl_v1,wl_v0,wl_min,wl_max):   
    # Create a polynomial representation with the proposed coefficients
    # Rescaling is done internally by providing the bounds l_min and l_max
    cheb = Chebyshev(poly_coeffs,[wl_min,wl_max])
    
    # Calculate the emissivities at the corresponding wavelengths
    eps1 = chebyshev.chebval(wl_v1,cheb.coef)
    eps0 = chebyshev.chebval(wl_v0,cheb.coef)

    # Invert of temperature    
    try:
        invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
    except:
        return 1e5
    
    # Temperature
    T = 1/invT
    T *= C2 * ( 1/wl_v1 - 1/wl_v0)
    
    ret = np.std(T)
    
    if np.isnan(ret):
        return 1e5
    else:   
        return ret
