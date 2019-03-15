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
from numpy.polynomial import Chebyshev,chebyshev

import matplotlib.pyplot as plt

from generate_spectrum import generate_data,wien_approximation
from pixel_operations import choose_pixels, generate_combinations
from temperature_functions import compute_temperature, compute_poly_temperature
from kfold import order_selection

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
T = 3000

#model_list = np.array([bb_eps,gr_eps,w_eps,art_eps])
model_list = np.array([w_eps,w_eps,w_eps,w_eps,w_eps,w_eps,w_eps,w_eps,w_eps,w_eps])

model_list = []
for it in range(2):
    model_list.append(w_eps)
    
model_list = np.array(model_list)

f,ax = plt.subplots(len(model_list),2)

it = 0

for f_eps in model_list:
    print("Model: ",it)
    
    ### Generate some data
    I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = generate_data(
            wl_vec,T,pix_vec,f_eps)
    wl_sub_vec = wl_vec[pix_sub_vec]
    

    
    ### Compute the temperature
#    Tave,std,rse,refined_fit,sol = compute_temperature(data_spl,cmb_pix,pix_sub_vec,wl_vec)
    with np.errstate(invalid='raise'):
        poly_order = order_selection(data_spl,filtered_data,
                           pix_sub_vec,wl_vec,
                           bb_eps)

        ### Pixel operations
        chosen_pix = choose_pixels(pix_sub_vec,bin_method='average')
        cmb_pix = generate_combinations(chosen_pix,pix_sub_vec)
        
        Tave,std,rse,sol = compute_poly_temperature(data_spl,cmb_pix,
                                                    pix_sub_vec,wl_vec,
                                                    poly_order)
        refined_fit = False
        if poly_order > 0:
            refined_fit = True

#    # Calculate the MSE
#    Ipred = wien_approximation(wl_sub_vec,Tave,bb_eps)
#    pwr = 0
#    eps_vec = np.zeros(len(wl_sub_vec))
#    for c in sol.x:
#        eps_vec += c * wl_sub_vec ** pwr
#        pwr += 1
#        
#    Ipred *= eps_vec
#    Ipred = np.log10(np.abs(Ipred))
#                
#    mse = 1/len(filtered_data) * np.sum((filtered_data - Ipred)**2)
#    

#        
#    
#    return Tave,std,rse,sol,mse,refined_fit
##    print(mse_array,np.nanmean(mse_array,axis=0))

    ### Reconstruct data
    bb_reconstructed = wien_approximation(wl_sub_vec,Tave,bb_eps)
    eps_vec_reconstructed = 10**filtered_data/bb_reconstructed
    # Since we get epsilon from the filtered data, "reconstructed_data" will be
    # exactly like "filtered_data"
        
    reconstructed_data = bb_reconstructed * eps_vec_reconstructed # exactly filtered   

    reconstruct_alt = wien_approximation(wl_sub_vec,Tave,bb_eps)
    pwr = 0
    eps_vec = np.zeros(len(wl_sub_vec))
    for c in sol.x:
        eps_vec += c * wl_sub_vec ** pwr
        pwr += 1
        
    reconstruct_alt *= eps_vec
#    reconstruct_alt = np.log10(np.abs(reconstruct_alt))    
    
    ### Plots
    if it == 0:
        ax[it][0].set_title("Intensity")
        ax[it][1].set_title("Emissivity")
        
    
    # Intensity
    ax[it][0].semilogy(wl_vec,noisy_data)
    ax[it][0].semilogy(wl_sub_vec,reconstructed_data)
    ax[it][0].semilogy(wl_sub_vec,reconstruct_alt)
    
    T_string = str(round(Tave,1)) + "+/-" + str(round(std,2)) + " K"
    error = np.abs((Tave-T)/T)*100
    
    T_string += "\n" + str(round(error,2)) + " %"
    ax[it][0].text(850,np.average(I_calc)/100,T_string)
    
    # Emissivity
    ax[it][1].plot(wl_vec,f_eps(wl_vec,Tave),'--')
    ax[it][1].plot(wl_sub_vec,eps_vec_reconstructed) 
    
    if not refined_fit:
        # Calculate the average emissivity
        eps_ave = np.average(eps_vec)
        eps_std = np.std(eps_vec)
        ax[it][1].plot(wl_vec,eps_ave*np.ones(len(wl_vec)))
        print(eps_ave,eps_std)
        
    if refined_fit:
        eps_poly = Chebyshev(sol.x,[np.min(wl_vec),np.max(wl_vec)])
        eps_val = chebyshev.chebval(wl_vec,eps_poly.coef)
        ax[it][1].plot(wl_vec,eps_val)
    

    
    it += 1
#
#
#ax[it-1][0].set_xlabel("Wavelength (nm)")
#ax[it-1][1].set_xlabel("Wavelength (nm)")
#plt.rcParams.update({'font.size':10})
#    
#    
#    ### Plots            


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

