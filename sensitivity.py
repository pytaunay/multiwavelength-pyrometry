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

import generate_spectrum as gs

from scipy.interpolate import splrep


from pixel_operations import choose_pixels, generate_combinations
from temperature_functions import optimum_temperature
from kfold import order_selection

import spectropyrometer_constants as sc

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
