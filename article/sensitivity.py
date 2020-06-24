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

from pixel_operations import choose_pixels, generate_combinations
from temperature_functions import optimum_temperature
from kfold import order_selection

import spectropyrometer_constant as sc

### Emissivity functions
# Black and gray body
bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
gr_eps = lambda wl,T: 0.1 * np.ones(len(wl))

### Vectors of pixels and wavelengths
wl_vec = np.linspace(400,800,(int)(3000))
pix_vec = np.linspace(0,2999,3000)
pix_vec = np.array(pix_vec,dtype=np.int64)

### Chosen temperature 
T = 1500

### Chosen emissivity function
chosen_eps = gr_eps

### Emission lines
#el = np.array([350,400,450,500,600,650,800])
el = None

### Iterate over multiple models
it = 0


### Generate some data
I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
        wl_vec,T,pix_vec,chosen_eps,el)
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

print(sc.window_length,Tave,Tstd,np.abs(Tave-T)/T)
