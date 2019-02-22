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
import matplotlib.pyplot as plt
import itertools as it
from scipy.interpolate import splrep,splev,UnivariateSpline
from scipy.stats import iqr
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit,minimize

C1 = 1.191e16 # W/nm4/cm2 Sr
C2 = 1.4384e7 # nm K

pix_slice = 300

'''
Function: wien_approx
Calculates the Wien approximation to Planck's law for non-constant emissivity
Inputs:
    - lnm: wavelength in nm
    - T: temperature in K
    - f_eps: a lambda function representing the emissivity as function of
    temperature and wavelength
'''
def wien_approx(lnm,T,f_eps):
    eps = f_eps(lnm,T) # Emissivity
    
    return eps * C1 / lnm**5 * np.exp(-C2/(T*lnm))

    
def min_multivariate(X,logR,l1,l0,lnm_vec):
    lnm_min = np.min(lnm_vec)
    lnm_max = np.max(lnm_vec)
    
    cheb = np.polynomial.Chebyshev(X,[lnm_min,lnm_max])
    
    eps1 = np.polynomial.chebyshev.chebval(l1,cheb.coef)
    eps0 = np.polynomial.chebyshev.chebval(l0,cheb.coef)

        
    invt = logR - 5 *np.log(l1/l0) - np.log(eps0/eps1)
    
    t = 1/invt
    t *= C2 * ( 1/l1 - 1/l0)
    
    ret = np.std(t)
    
#    ### Also calculate deviation from the data curve
#    bb_eps = lambda lnm,T: 1.0 # Black body
#    Tave = np.average(t)
#    epsvec = eps_piecewise(X,lnm_vec,lnm_binm,lnm_binM)
#    test_curve = epsvec*wien_approx(lnm_vec,Tave,bb_eps)
#    
#    stdcurve = np.std(np.abs(filtered_data-test_curve))
    
    if np.isnan(ret):
        return 1e5
    else:   
        return ret
    

def T_multivariate(X,logR,l1,l0,lnm_binm,lnm_binM):  
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

### Emissivity functions
# Tungsten 2000 K emissivity and polynomial of order 1 to fit it
w_lnm = np.array([300,350,400,500,600,700,800,900])
w_eps_data = np.array([0.474,0.473,0.474,0.462,0.448,0.436,0.419,0.401])

w_m,w_b = np.polyfit(w_lnm,w_eps_data,deg=1)
w_eps = lambda lnm,T: w_m*lnm + w_b

# Black and gray body
bb_eps = lambda lnm,T: 1.0 * np.ones(len(lnm))
gr_eps = lambda lnm,T: 0.1 * np.ones(len(lnm))

# Artificial tests
art_lnm = np.array([300,500,1100])
art_eps_data = np.array([1,0.3,1])
art_fac = np.polyfit(art_lnm,art_eps_data,deg=2)

a0,a1,a2 = art_fac
art_eps = lambda lnm,T: a0*lnm**2 + a1*lnm + a2

### Vectors of pixels and wavelengths
lnm_vec = np.linspace(300,1100,(int)(3000))
pix_vec = np.linspace(0,2999,3000)
pix_vec = np.array(pix_vec,dtype=np.int64)

### Chosen emissivity function
chosen_eps = w_eps

### Create data
T = 2000
print(T)

# Intensity with addeded noise
I_calc = wien_approx(lnm_vec,T,chosen_eps)
noisy_data = np.random.normal(I_calc,0.1*I_calc)

### Filter data with a Savgol filter
window_length = (int)(pix_slice/5)
if window_length % 2 == 0:
    window_length += 1

filtered_data = savgol_filter(noisy_data,window_length,3)

### Fit a line through the noise with some smoothing
spl = splrep(lnm_vec,np.log10(filtered_data))

### Bin the pixels
bins = pix_vec[0::pix_slice]

### For each bin, find a pixel
rand_pix = []
for idx in bins[:-1]:
    # Find the low and high bounds of that slice
    lo = pix_vec[idx]
    hi = pix_vec[idx + pix_slice - 1]
    
    # Pick a pixel at random in it
#    pix_idx = np.random.choice(pix_vec[lo:hi],size=1)[0]
    pix_idx = (int)(np.average([lo,hi]))
    
    rand_pix.append(pix_idx)
    
rand_pix = np.array(rand_pix)

Tout = []
c_pix_array = []
logR_array = []

### For each pixel p0 (i.e. rpix)...
for rpix in rand_pix:
    p0 = rpix
    
    # Get the other pixels above and below this pixel rpix
    p1vec_p = pix_vec[rpix::pix_slice]
    p1vec_m = pix_vec[rpix::-pix_slice]
    
    # Create a vector of pixels, remove any duplicates, make sure we do not
    # include rpix
    p1vec = np.concatenate((p1vec_m,p1vec_p))
    p1vec = np.unique(p1vec)
    p1vec = p1vec[p1vec != rpix]
    
    # Calculate the gray body temperature predicted by each pair of pixels         
    for p1 in p1vec:      
        # Pixels to wavelength          
        l0 = lnm_vec[p0]
        l1 = lnm_vec[p1]
        
        # Corresponding data from the filtered data
        res0 = 10**splev(l0,spl)
        res1 = 10**splev(l1,spl)
      
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

### Exclude data w/ Tukey fencing
T_iqr = iqr(Tout)
T_qua = np.percentile(Tout,[25,75])

min_T = T_qua[0] - 1.25*T_iqr
max_T = T_qua[1] + 1.25*T_iqr

T_left = Tout[(Tout>min_T) & (Tout<max_T)]

### Calculate standard deviation, average, standard error
std = np.std(T_left)
Tave = np.mean(T_left)
rse = std/Tave

print(Tave,std,rse*100)

refined_fit = False

### Do we have a "good enough" fit?   
# If not, a few more operations are required
if rse*100 > 0.75:
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
    X0 = np.zeros(3)
    X0[0] = 1

    min_options = {'xatol':1e-15,'fatol':1e-15,'maxfev':3000}
    sol = minimize(f,X0,method='Nelder-Mead',options = min_options)
#    sol = minimize(f,X0,options=min_options)

    Tave,std = T_multivariate(sol.x,logR_array,lnm_vec1,lnm_vec0,lnm_binm,lnm_binM)
    print(Tave,std,std/Tave*100,sol.x)

### Plots            
bb_reconstructed = wien_approx(lnm_vec,Tave,bb_eps)
eps_vec = filtered_data/bb_reconstructed
# Since we get epsilon from the filtered data, "reconstructed_data" will be
# exactly like "filtered_data"
reconstructed_data = bb_reconstructed * eps_vec # exactly filtered
    
## Subplots
f, (ax1, ax2) = plt.subplots(1, 2)
# Plot the intensity
ax1.semilogy(lnm_vec,noisy_data)
ax1.semilogy(lnm_vec,reconstructed_data)

# Plot the emissivity
ax2.plot(lnm_vec,eps_vec)
ax2.plot(lnm_vec,chosen_eps(lnm_vec,Tave),'--')

if refined_fit:
    eps_poly = np.polynomial.Chebyshev(sol.x,[np.min(lnm_vec),np.max(lnm_vec)])
    eps_val = np.polynomial.chebyshev.chebval(lnm_vec,eps_poly.coef)
    
    ax2.plot(lnm_vec,eps_val,'-.')

#epsret = eps_piecewise(sol.x,lnm_vec,lnm_binm,lnm_binM)
#ax2.plot(lnm_vec,epsret,'-.')

