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

from generate_spectrum import generate_data
from pixel_operations import choose_pixels, generate_combinations
from temperature_functions import compute_temperature

### Emissivity functions
# Tungsten 2000 K emissivity and polynomial of order 1 to fit it
w_wl = np.array([300,350,400,500,600,700,800,900])
w_eps_data = np.array([0.474,0.473,0.474,0.462,0.448,0.436,0.419,0.401])

w_m,w_b = np.polyfit(w_wl,w_eps_data,deg=1)
w_eps = lambda wl,T: w_m*wl + w_b

# Black and gray body
bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
gr_eps = lambda wl,T: 0.1 * np.ones(len(wl))

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

### Chosen emissivity function
chosen_eps = gr_eps

### Create data
T = 2000

model_list = np.array([bb_eps,gr_eps,w_eps,art_eps])

f,ax = plt.subplots(len(model_list),2)

it = 0

for f_eps in model_list:
    print("Model: ",it)
    
    # Generate some data
    I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = generate_data(
            wl_vec,T,pix_vec,f_eps)
    wl_sub_vec = wl_vec[pix_sub_vec]
    
    # Pixel operations
    chosen_pix = choose_pixels(pix_sub_vec,bin_method='average')
    cmb_pix = generate_combinations(chosen_pix,pix_sub_vec)
    
    # Compute the temperature
    compute_temperature(data_spl,cmb_pix,pix_sub_vec,wl_vec)
    
    it += 1

    ### Plots

#    
#    
#    ### Plots            
#    bb_reconstructed = wien_approx(lnm_vec_sub,Tave,bb_eps)
#    eps_vec = 10**filtered_data/bb_reconstructed
#    # Since we get epsilon from the filtered data, "reconstructed_data" will be
#    # exactly like "filtered_data"
#    reconstructed_data = bb_reconstructed * eps_vec # exactly filtered
#        
#    ## Subplots
#    f, (ax1, ax2) = plt.subplots(1, 2)
#    # Plot the intensity
#    ax1.semilogy(lnm_vec,noisy_data)
#    ax1.semilogy(lnm_vec_sub,reconstructed_data)
#    #ax1.semilogy(lnm_vec,)
#    
#    # Plot the emissivity
#    ax2.plot(lnm_vec_sub,eps_vec)
#    ax2.plot(lnm_vec,chosen_eps(lnm_vec,Tave),'--')
#    
#    if refined_fit:
#        eps_poly = np.polynomial.Chebyshev(sol.x,[np.min(lnm_vec),np.max(lnm_vec)])
#        eps_val = np.polynomial.chebyshev.chebval(lnm_vec,eps_poly.coef)
#        
#        ax2.plot(lnm_vec,eps_val,'-.')
#    
#    #epsret = eps_piecewise(sol.x,lnm_vec,lnm_binm,lnm_binM)
#    #ax2.plot(lnm_vec,epsret,'-.')

