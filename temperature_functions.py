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

from scipy.interpolate import splev
from scipy.stats import iqr

from spectropyrometer_constants import pix_slice,C2

'''
Function: tukey_fence
Removes outliers using Tukey fencing
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
    
    T_left = Tout[(Tout>min_T) & (Tout<max_T)]
    
    ### Calculate standard deviation, average, standard error
    std = np.std(T_left)
    Tave = np.mean(T_left)
    rse = std/Tave
    
    print(Tave,std,rse*100)    

'''
Function: simple_temperature
Calculates the temperature based on the averaging of multiple two-temperature
predictions
Inputs:
    - data_spl Spline representation of the filtered intensity data
    - pix_binned Pixels chosen for each pixel bin
    - pix_vec Overall pixel vector
    - wl_vec Vector of wavelengths (nm)
Ouputs:
    - Predicted temperature from averaging (K)
    - Standard deviation (K)
    - Standard deviation (%)
    - Flag indicating if advanced method was used
'''
def simple_temperature(data_spl,pix_binned,pix_vec,wl_vec):
    Tout = []
    c_pix_array = []
    logR_array = []
    
    ### Standard computation
    # For each pixel p0...
    for p0 in pix_binned:
        
        # Get corresponding pair pixel above and below this pixel p0
        # They belong to other slices
        p1vec_p = pix_vec[p0::pix_slice]
        p1vec_m = pix_vec[p0::-pix_slice]
        
        # Create a vector of pixels, remove any duplicates, make sure we do not
        # include p0
        p1vec = np.concatenate((p1vec_m,p1vec_p))
        p1vec = np.unique(p1vec)
        p1vec = p1vec[p1vec != p0]
        
        # Calculate the gray body temperature predicted by each pair of pixels         
        for p1 in p1vec:      
            # Pixels to wavelength          
            l0 = wl_vec[p0]
            l1 = wl_vec[p1]
            
            # Corresponding data from the filtered data
            res0 = 10**splev(l0,data_spl)
            res1 = 10**splev(l1,data_spl)
          
            # Ratio of intensities
            R = res0/res1
            
            # Handle edge cases
            # Try/catch to make sure the log spits out correct values
            try:
                Ttarget = C2 * ( 1/l1 - 1/l0) / (np.log(R)-5*np.log(l1/l0))
            except:
                continue
            
            # Skip if negative or NaN
            if Ttarget < 0 or np.isnan(Ttarget):
                continue
            
            # Build vector
            Tout.append(Ttarget)
            c_pix_array.append((p0,p1))
            logR_array.append(np.log(R))
    
    ### Convert to numpy arrays
    c_pix_array = np.array(c_pix_array)
    Tout = np.array(Tout)      
    logR_array = np.array(logR_array)    




'''
Function: compute_temperature
Calculates the temperature
Inputs:
    - data_spl Spline representation of the filtered intensity data
    - pix_binned Pixels chosen for each pixel bin
    - pix_vec Overall pixel vector
    - wl_vec Vector of wavelengths (nm)
Ouputs:
    - Predicted temperature from averaging (K)
    - Standard deviation (K)
    - Standard deviation (%)
    - Flag indicating if advanced method was used
'''
def compute_temperature(data_spl,pix_binned,pix_vec,wl_vec):

    






def compute_T(X,logR,l1,l0,lnm_binm,lnm_binM):  
    lnm_min = np.min(lnm_vec)
    lnm_max = np.max(lnm_vec)
    
    cheb = np.polynomial.Chebyshev(X,[lnm_min,lnm_max])
    
    eps1 = np.polynomial.chebyshev.chebval(l1,cheb.coef)
    eps0 = np.polynomial.chebyshev.chebval(l0,cheb.coef)
    
    invt = logR - 5 *np.log(l1/l0) - np.log(eps0/eps1)
#    print(invt)
    
    Tout = 1/invt
    Tout *= C2 * ( 1/l1 - 1/l0)

    T_iqr = iqr(Tout)
    T_qua = np.percentile(Tout,[25,75])
        
    min_T = T_qua[0] - 1.25*T_iqr
    max_T = T_qua[1] + 1.25*T_iqr
        
    T_left = Tout[(Tout>min_T) & (Tout<max_T)]
    
    ret = np.average(T_left)
    std = np.std(T_left)
    
    return ret,std    
    


refined_fit = False

### Do we have a "good enough" fit?   
# If not, a few more operations are required
nunk = 1
while rse*100 > 0.5:
    refined_fit = True
    
    # Which wavelengths are associated with the pixel combinations?
    lnm_vec0 = lnm_vec[c_pix_array[:,0]]
    lnm_vec1 = lnm_vec[c_pix_array[:,1]]    

    # Create the [lambda_min,lambda_max] pairs that delimit a "bin"
    lnm_binm = lnm_vec[bins]
    lnm_binM = lnm_vec[bins[1::]]
    lnm_binM = np.append(lnm_binM,lnm_vec[-1])
    
    f = lambda X: min_multivariate(X,logR_array,lnm_vec1,lnm_vec0,lnm_vec)
    
    # Chebyshev polynomial coefficients
#    X0 = np.zeros(len(lnm_binm))
#    X0 = np.zeros(3)
    X0 = np.zeros(nunk)
    X0[0] = 0.1

    min_options = {'xatol':1e-10,'fatol':1e-10,'maxfev':5000}
    sol = minimize(f,X0,method='Nelder-Mead',options = min_options)

    Tave,std = T_multivariate(sol.x,logR_array,lnm_vec1,lnm_vec0,lnm_binm,lnm_binM)
    print(Tave,std,std/Tave*100,sol.x)
    
    rse = std/Tave
    nunk = nunk + 1