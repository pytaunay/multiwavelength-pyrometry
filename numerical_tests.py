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

import numpy as np
from numpy.polynomial import Polynomial, polynomial

import matplotlib.pyplot as plt
import generate_spectrum as gs

from pixel_operations import choose_pixels, generate_combinations
from temperature_functions import optimum_temperature
from kfold import order_selection

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
    
    elif chosen_case =='black_body':
        # Black body
        f_eps = lambda wl,T: 1.0 * np.ones(len(wl))
    elif chosen_case == 'gray_body':
        # Gray body
        f_eps = lambda wl,T: 0.5 * np.ones(len(wl))
    elif chosen_case == 'second_order':
        # Artificial second order
        art_wl = np.array([300,500,1100])
        art_eps_data = np.array([1,0.3,1])
        art_fac = np.polyfit(art_wl,art_eps_data,deg=2)
        
        a0,a1,a2 = art_fac
        f_eps = lambda wl,T: a0*wl**2 + a1*wl + a2
    else:
        # If none of the valid case are correct, throw a runtime error.
        # This should not happen but one is never too careful.
        raise RuntimeError("Invalid chosen case") 

    return f_eps

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

## Temperature
T0 = 1500

### Run
if chosen_case is not 'gray_body' or 'tungsten' or 'second_order':
    raise RuntimeError("Invalid chosen case")

f_eps = select_true_emissivity(chosen_case)

# Vectors of pixels and wavelengths
wl_vec = np.linspace(wl_min,wl_max,(int)(npix))
pix_vec = np.linspace(0,npix-1,npix,dtype=np.int64)

#### Chosen emissivity function
#chosen_eps = art_eps
#model_list = []
#for it in range(2):
#    model_list.append(chosen_eps)
#
#model_list = np.array(model_list)

### Emission lines
#el = np.array([350,400,450,500,600,650,800])
el = None

### Plots
f,ax = plt.subplots(len(model_list),2)

### Iterate over multiple models
it = 0
for f_eps in model_list:
    print("Model: ", it)

    ### Generate some data
    I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
            wl_vec,T,pix_vec,f_eps,el)
    wl_sub_vec = wl_vec[pix_sub_vec]
    

    ### Choose the order of the emissivity w/ k-fold
    poly_order = order_selection(data_spl,filtered_data,
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
    eps_vec_reconstructed = 10**filtered_data/bb_reconstructed
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
    ax[it][0].semilogy(wl_sub_vec,reconstructed_alt)

    T_string = str(round(Tave,1)) + "+/-" + str(round(Tstd,2)) + " K"
    error = np.abs((Tave-T)/T)*100

    T_string += "\n" + str(round(error,2)) + " %"
    ax[it][0].text(850,np.average(I_calc)/100,T_string)

    # Emissivity
    ax[it][1].plot(wl_vec,f_eps(wl_vec,Tave),'--')
    ax[it][1].plot(wl_sub_vec,eps_vec_reconstructed) 
    ax[it][1].plot(wl_sub_vec,eps_vec,'-.')

    it += 1
