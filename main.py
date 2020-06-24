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
from numpy.polynomial import Polynomial, polynomial

import matplotlib.pyplot as plt

import algorithm.generate_spectrum as gs

from algorithm.pixel_operations import choose_pixels, generate_combinations
from algorithm.temperature_functions import optimum_temperature
from algorithm.kfold import order_selection

### Emissivity functions
# Tungsten 2000 K emissivity and polynomial of order 1 to fit it
w_wl = np.array([300,350,400,500,600,700,800,900])
w_eps_data = np.array([0.474,0.473,0.474,0.462,0.448,0.436,0.419,0.401])

w_m,w_b = np.polyfit(w_wl,w_eps_data,deg=1)
w_eps = lambda wl,T: w_m*wl + w_b

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
wl_vec = np.linspace(300,1100,(int)(3000))
pix_vec = np.linspace(0,2999,3000)
pix_vec = np.array(pix_vec,dtype=np.int64)

### Chosen temperature 
T_list = np.array([1500.,2000.,3000.])

### Chosen emissivity function
chosen_eps = art_eps
model_list = [gr_eps,w_eps,art_eps]
model_list = np.array(model_list)

### Emission lines
#el = np.array([350,400,450,500,600,650,800])
el = None

### Plots
f,ax = plt.subplots(len(model_list),2)

### Iterate over multiple models
it = 0
for T,f_eps in zip(T_list,model_list):
    print("Model: ", it)

    ### Generate some data
    I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
            wl_vec,T,pix_vec,f_eps,el)
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


    ### Plots
    if it == 0:
        ax[it][0].set_title("Intensity")
        ax[it][1].set_title("Emissivity")

    # Intensity
    ax[it][0].semilogy(wl_vec,noisy_data)
    ax[it][0].semilogy(wl_sub_vec,reconstructed_data)

    T_string = str(round(Tave,1)) + "+/-" + str(round(Tstd,2)) + " K"
    error = np.abs((Tave-T)/T)*100

    T_string += "\n" + str(round(error,2)) + " %"
    ax[it][0].text(850,np.average(I_calc)/100,T_string)

    # Emissivity
    ax[it][1].plot(wl_vec,f_eps(wl_vec,Tave),'--')
    ax[it][1].plot(wl_sub_vec,eps_vec_reconstructed) 

    it += 1
