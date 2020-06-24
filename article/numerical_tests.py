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

'''
File: numerical_tests.py

Description: select a test-emissivity and temperature and run the algorithm on 
the artificially generated radiance to find the original emissivity and 
temperature.

It will generate plots that are similar to Fig. 9 in our original paper.
'''

import numpy as np
from numpy.polynomial import Polynomial, polynomial

import matplotlib.pyplot as plt
import algorithm.generate_spectrum as gs

from algorithm.pixel_operations import choose_pixels, generate_combinations
from algorithm.temperature_functions import optimum_temperature
from algorithm.kfold import order_selection

def select_true_emissivity(chosen_case):
    '''
    Returns the true emissivity based on the case of interest chosen.
    Inputs:
        - chosen_case: case of interest
    Returns:
        - lambda function for the case of interest
    '''
    ### Emissivity functions
    if chosen_case == 'tungsten':   
        # Tungsten 2000 K emissivity and polynomial of order 1 to fit it
        # Source: CRC Handbook
        w_wl = np.array([300,350,400,500,600,700,800,900])
        w_eps_data = np.array([0.474,0.473,0.474,0.462,0.448,0.436,0.419,0.401])
    
        w_m,w_b = np.polyfit(w_wl,w_eps_data,deg=1)
        f_eps = lambda wl,T: w_m*wl + w_b
        T0 = 2000
    
    elif chosen_case =='black_body':
        # Black body
        f_eps = lambda wl,T: 1.0 * np.ones(len(wl))
        T0 = 1500
    elif chosen_case == 'gray_body':
        # Gray body
        f_eps = lambda wl,T: 0.5 * np.ones(len(wl))
        T0 = 1500
    elif chosen_case == 'second_order':
        # Artificial second order
        art_wl = np.array([300,500,1100])
        art_eps_data = np.array([1,0.3,1])
        art_fac = np.polyfit(art_wl,art_eps_data,deg=2)
        
        a0,a1,a2 = art_fac
        f_eps = lambda wl,T: a0*wl**2 + a1*wl + a2
        T0 = 3000
    else:
        # If none of the valid case are correct, throw a runtime error.
        # This should not happen but one is never too careful.
        raise RuntimeError("Invalid chosen case") 

    return f_eps, T0

### Controls
## Case  of interset. chosen_case can be
# - "black_body"
# - "gray_body"
# - "tungsten"
# - "second_order"
chosen_case = 'gray_body'

## Wavelength range
wl_min = 400
wl_max = 800

## Number of CCD pixels
npix = 3000

### Run
bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))

if chosen_case != 'gray_body'  and chosen_case != 'tungsten' and chosen_case != 'second_order':
    raise RuntimeError("Invalid chosen case")

f_eps, T0 = select_true_emissivity(chosen_case)

# Vectors of pixels and wavelengths
wl_vec = np.linspace(wl_min,wl_max,(int)(npix))
pix_vec = np.linspace(0,npix-1,npix,dtype=np.int64)

### Emission lines
#el = np.array([350,400,450,500,600,650,800])
#el = None

### Plots
f,ax = plt.subplots(1,2)

### Iterate over multiple models
### Generate some data
# I_calc -> true radiance
# noisy_data -> perturbed radiance
# filtered_data -> averaged log(noisy_data)
# data_spl -> spline representation of filtered_data (no smoothing)
# pix_sub_vec -> pixels numbers used to address the main wavelength vector
# wl_vec -> main wavelength vector
# wl_sub_vec -> subset of the main wavelength vector
I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
        wl_vec,T0,pix_vec,f_eps)
wl_sub_vec = wl_vec[pix_sub_vec]


### Choose the order of the emissivity w/ k-fold
poly_order = order_selection(data_spl,
                   pix_sub_vec,wl_vec,
                   bb_eps)

### Calculate the temperature using the whole dataset
# Pixel operations
chosen_pix = choose_pixels(pix_sub_vec,bin_method='average')
cmb_pix = generate_combinations(chosen_pix,pix_sub_vec)

# Compute the temperature
Tave, Tstd, Tmetric, sol = optimum_temperature(data_spl,cmb_pix,
                                            pix_sub_vec,wl_vec,
                                            poly_order)

### Reconstruct data
# Black-body radiance based on the calculated temperature
bb_reconstructed = gs.wien_approximation(wl_sub_vec,Tave,bb_eps)
# Emissivity is calculated from the filtered data
eps_vec_reconstructed = np.exp(filtered_data)/bb_reconstructed
# Since we get epsilon from the filtered data, "reconstructed_data" will be
# exactly like "filtered_data"
reconstructed_data = bb_reconstructed * eps_vec_reconstructed 

# Alternative using the polynomial from optimization
reconstructed_alt = gs.wien_approximation(wl_sub_vec,Tave,bb_eps)
wl_min = np.min(wl_sub_vec)
wl_max = np.max(wl_sub_vec)

# If we found a polynomial emissivity, calculate its numerical values
if poly_order > 0:
    pol = Polynomial(sol.x,[wl_min,wl_max])
    eps_vec = polynomial.polyval(wl_sub_vec,pol.coef)
# If the emissivity model is constant, average its value over the 
# wavelengths
else:
    eps_ave = np.average(eps_vec_reconstructed)
    eps_vec = eps_ave * np.ones(len(wl_sub_vec))

# The "alternative" reconstruction where we use the optimized function
# It results in radiances that are offset by a multiplicative constant    
reconstructed_alt *= eps_vec

### Plots
ax[0].semilogy(wl_vec[0::99],1.3*I_calc[0::99],'k:')
ax[0].semilogy(wl_vec[0::99],0.7*I_calc[0::99],'k:')
ax[0].semilogy(wl_vec[0::99],I_calc[0::99],'k-')
        
ax[0].semilogy(wl_sub_vec,reconstructed_data)

T_string = str(round(Tave,1)) + " K"
error = np.abs((Tave-T0)/T0)*100

T_string += "\n" + str(round(error,2)) + " %"
ax[0].text(0.8*wl_max,np.average(I_calc)/100,T_string)

# Emissivity
ax[1].plot(wl_vec,f_eps(wl_vec,Tave),'k-')
ax[1].plot(wl_sub_vec,eps_vec_reconstructed) 
ax[1].set_ylim([0,1])

