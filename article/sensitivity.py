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
File: sensitivity.py

Description: perform sensitivity analysis on the computational threshold
and the smoothing filter window length. The threshold must be changed
by hand in the file algorithm/spectropyrometer_constants.py

The results may be used to generate Fig. 10 in our 2020 RSI Journal article
'''

import numpy as np

import algorithm.generate_spectrum as gs

from scipy.interpolate import splrep

from algorithm.pixel_operations import choose_pixels, generate_combinations
from algorithm.temperature_functions import optimum_temperature
from algorithm.kfold import order_selection

import algorithm.spectropyrometer_constants as sc

### Emissivity functions
# Black and gray body
bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
gr_eps = lambda wl,T: 0.5 * np.ones(len(wl))

# Artificial tests
art_wl = np.array([300,500,1100])
art_eps_data = np.array([1,0.3,1])
art_fac = np.polyfit(art_wl,art_eps_data,deg=2)

a0,a1,a2 = art_fac
art_eps = lambda wl,T: a0*wl**2 + a1*wl + a2

### Vectors of pixels and wavelengths
wl_vec = np.linspace(400,800,(int)(3000))
pix_vec = np.linspace(0,2999,3000)
pix_vec = np.array(pix_vec,dtype=np.int64)

### Chosen temperature 
T = 1500

### Chosen emissivity function
chosen_eps = art_eps

### Emission lines
#el = np.array([350,400,450,500,600,650,800])
el = None

### Iterate over multiple models
it = 0

### Windows
#wdw_array = np.array([3,5,7,9,11,21,31,41,51,61,71,81,91,101,201,301,401,501])
#wdw_array = np.array([31,41,51,61,71,81,91,101,201,301,401,501])
#wdw_array = np.array([61,71,81,91,101,201,301,401,501])
#wdw_array = np.array([101,201,301,401,501])
wdw_array = np.array([501])

# Intensity from Wien's approximation: true data
I_calc = gs.wien_approximation(wl_vec,T,chosen_eps)

### Generate some data
for wdw in wdw_array:

    
    # Add some noise and take the log of the data
    noisy_data = np.random.normal(I_calc,0.1*I_calc)
    log_noisy = np.log(noisy_data)
    
    # Remove the peaks
    nopeak = np.copy(log_noisy)  
    
    # Moving average filter
    wl = wdw
    log_med = gs.moving_average(nopeak,wl)

    
    ### Remove the edge effects
    if wl > 1:
        wl_vec_sub = wl_vec[wl-1:-(wl-1)]
        log_med = log_med[(int)((wl-1)/2):-(int)((wl-1)/2)]
        pix_vec_sub = pix_vec[wl-1:-(wl-1)]
    else:
        wl_vec_sub = np.copy(wl_vec)
        pix_vec_sub = np.copy(pix_vec)
            
    ### Fit a spline to access data easily
    data_spl = splrep(wl_vec_sub,log_med)


    filtered_data = np.copy(log_med)
    pix_sub_vec = np.copy(pix_vec_sub)


    wl_sub_vec = wl_vec[pix_sub_vec]

    
   
    ntests = 1000
    errvec = []
    stdvec = []
    for idx in range(ntests):
        if np.mod(idx,ntests/10) == 0:
            print(idx)
        
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
    
        err = np.abs(Tave-T)/T * 100
        
        errvec.append(err)
        stdvec.append(Tstd)
    
    errvec = np.array(errvec)
    stdvec = np.array(stdvec)
    print(sc.threshold, wdw, np.mean(errvec), np.mean(stdvec))
