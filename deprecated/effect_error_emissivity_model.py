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

import matplotlib.pyplot as plt
import algorithm.generate_spectrum as gs

from algorithm.pixel_operations import choose_pixels, generate_combinations
from algorithm.temperature_functions import optimum_temperature
from algorithm.kfold import order_selection

'''
File: effect_error_emissivity_model.py
Description: investigate the effect of an erroneous emissivity model on the
error and coefficient of dispersion.

The output data is used to generate Fig. 6 in our RSI journal article.
'''

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

#### Chosen emissivity function
model_list = []
for it in range(100):
    model_list.append(f_eps)

model_list = np.array(model_list)

### Emission lines
#el = np.array([350,400,450,500,600,650,800])
#el = None

### Plots
#f,ax = plt.subplots(len(model_list),2)

### Iterate over multiple models
it = 0
polyvec = []
Tavevec = []
errvec = []

for idx,f_eps in enumerate(model_list):
    if np.mod(idx,10) == 0:
        print(idx)
    
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

    ### Store data
    err = (Tave/T0-1)*100
    polyvec.append(poly_order)
    Tavevec.append(Tave)
    errvec.append(err)
    
    print(idx,err,poly_order,Tave)
    
    
print(np.mean(np.abs(errvec)),
      np.mean(polyvec),
      np.mean(Tavevec))
