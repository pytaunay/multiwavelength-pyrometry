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
File: duvaut1995.py
Description: Apply the algorithm on artificially generated radiance (with 
experimentally measured emissivity). The goal is to find emissivity and 
temperature of a tantalum surface which is held at 573 K.

The experimental emissivity is given in the following reference:
[1] T. Duvaut, D. Georgeault, and J. L. Beaudoin, "Multiwavelength infrared 
pyrometry: optimization and computer simulations," Infrared Physics and 
Technology, 36, 7, pp. 1089-1103, 1995.
'''

import numpy as np
from numpy.polynomial import Polynomial, polynomial

import matplotlib.pyplot as plt
import algorithm.generate_spectrum as gs

from algorithm.pixel_operations import choose_pixels, generate_combinations
from algorithm.temperature_functions import optimum_temperature
from algorithm.kfold import order_selection

from scipy.interpolate import splrep, splev

### Emissivity functions
# Black and gray body
bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
gr_eps = lambda wl,T: 0.1 * np.ones(len(wl))

# Emissivity from Duvaut; same method: generate radiance from emissivity
data = np.genfromtxt('data/duvaut-1995/tantalum-emissivity.csv', delimiter=',',skip_header=1)
eps_xp = data[:,1]
wl_eps_xp = data[:,0] * 1000 # Wavelengths are in micro-meter
T = 573
Ttrue = T

# Spline the data to generate the "true" emissivity
eps_spl = splrep(wl_eps_xp,eps_xp)
f_eps = lambda wl,T: splev(wl,eps_spl)

#### Vectors of pixels and wavelengths
wl_vec = np.linspace(np.min(wl_eps_xp),np.max(wl_eps_xp),(int)(3000))
pix_vec = np.linspace(0,2999,3000)
pix_vec = np.array(pix_vec,dtype=np.int64)


### Generate some data
I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
        wl_vec,T,pix_vec,f_eps)
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
bb_reconstructed = gs.wien_approximation(wl_sub_vec,Tave,bb_eps)
eps_vec_reconstructed = np.exp(filtered_data)/bb_reconstructed
# Since we get epsilon from the filtered data, "reconstructed_data" will be
# exactly like "filtered_data"
reconstructed_data = bb_reconstructed * eps_vec_reconstructed # exactly filtered   

# Alternative using the polynomial from optimization
reconstructed_alt = gs.wien_approximation(wl_sub_vec,Tave,bb_eps)
wl_min = np.min(wl_sub_vec)
wl_max = np.max(wl_sub_vec)

if poly_order > 0:
    cheb = Polynomial(sol.x,[wl_min,wl_max])
    eps_vec = polynomial.polyval(wl_sub_vec,cheb.coef)
    
else:
    eps_ave = np.average(eps_vec_reconstructed)
    eps_vec = eps_ave * np.ones(len(wl_sub_vec))
    
reconstructed_alt *= eps_vec


#### Plots
data = np.genfromtxt('data/duvaut-1995/tantalum-irradiance.csv', delimiter=',',skip_header=1)
radiance = data[:,1] / 1e7
wl_vec_radiance = data[:,0] * 1000 # Wavelengths are in micro-meter

fig, ax = plt.subplots(2,1)
ax[0].semilogy(wl_vec_radiance,radiance)
ax[0].semilogy(wl_vec,noisy_data)
ax[0].semilogy(wl_sub_vec,reconstructed_data)
ax[0].set_title("Radiance")

ax[1].plot(wl_eps_xp,eps_xp)
ax[1].plot(wl_sub_vec,eps_vec_reconstructed)
ax[1].set_title("Emissivity")

error = np.abs((Tave-Ttrue)/Ttrue)*100
print(Ttrue,error)

