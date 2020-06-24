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
import algorithm.spectropyrometer_constants as sc

from numpy.polynomial import Polynomial, polynomial
from algorithm.statistics import tukey_fence

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
#            T = T[T>0]
            Tave, Tstd, Tmetric, _ = tukey_fence(T)
#            print(Tave,Tmetric)
            
            if Tave < 0 or Tmetric < 0:
                raise
            
            ret = Tmetric
            
        except:
            ret = 1e5
    
    if np.isnan(ret):
        return 1e5
    else:   
        return ret
