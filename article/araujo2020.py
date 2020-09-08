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

from scipy.interpolate import splrep

def araujo_emissivity(esp,espp,c,r,lmin,lmax,wl):
    xi = r/(lmax-lmin)
    
    l0 = (1-c) * lmin + c * lmax
    
    xp = 1 + np.exp(-xi * (lmin-l0))
    xpp = 1 + np.exp(-xi * (lmax-l0))
    
    deps = (espp-esp) / (1/xpp-1/xp)
    eps0 = (xpp * espp - xp * esp)  / (xpp-xp)
    
    eps = eps0 + deps / (1+np.exp(-xi*(wl-l0)))
    
    return eps
        

# Black and gray body
bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
gr_eps = lambda wl,T: 0.5 * np.ones(len(wl))

### Generate some data
T = 1000 # K

# First create the emissivity
lambda_min = 0.4 * 1e3
lambda_max = 0.8 * 1e3

# Parameters
Nwl = 3000 # 3000 wavelengths
wl_vec = np.linspace(lambda_min,lambda_max,Nwl)

pix_vec = np.linspace(0,len(wl_vec)-1,len(wl_vec))
pix_vec = np.array(pix_vec,dtype=np.int64)

# Radiance
eps1 = lambda wl,T: araujo_emissivity(0.3,0.9,1,4,lambda_min,lambda_max,wl)
eps2 = lambda wl,T: araujo_emissivity(0.9,0.3,0.7,20,lambda_min,lambda_max,wl)

noise = True

if noise:
    I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
            wl_vec,T,pix_vec,eps2)
    
#    print("Already calculated")
else:
    I_calc = gs.wien_approximation(wl_vec,T,eps2)
    noisy_data = np.copy(I_calc)
    
    # Take the log of the data
    
    log_noisy = np.log(noisy_data)
    
    wl_vec_sub = np.copy(wl_vec)
    log_med = np.copy(log_noisy)
    pix_vec_sub = np.copy(pix_vec)
    
    ### Fit a spline to access data easily
    data_spl = splrep(wl_vec_sub,log_med)
    
    pix_sub_vec = np.copy(pix_vec_sub)
    filtered_data = np.copy(log_med)


#I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
#        wl_vec,T,pix_vec,f_eps,el)
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
#eps_vec_reconstructed = 10**filtered_data/bb_reconstructed
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
fig, ax = plt.subplots(2,1)
ax[0].semilogy(wl_vec,noisy_data)
ax[0].semilogy(wl_sub_vec,reconstructed_data)
#ax[0].semilogy(wl_sub_vec,reconstructed_alt)

#data = np.genfromtxt('data/wen-2011-emissivity.csv', delimiter=',',skip_header=1)
#eps_xp = data[:,1]
#wl_eps_xp = data[:,0] * 1000 # Wavelengths are in micro-meter

ax[1].plot(wl_vec,eps2(wl_vec,T))
#ax[1].plot(wl_vec,eps1(wl_vec,T))
ax[1].plot(wl_sub_vec,eps_vec_reconstructed)

#if it == 0:
#    ax[it][0].set_title("Intensity")
#    ax[it][1].set_title("Emissivity")
#
## Intensity
#ax[it][0].semilogy(wl_vec,noisy_data)
#ax[it][0].semilogy(wl_sub_vec,reconstructed_data)
#ax[it][0].semilogy(wl_sub_vec,reconstructed_alt)
#
#T_string = str(round(Tave,1)) + "+/-" + str(round(Tstd,2)) + " K"
error = np.abs((Tave-T)/T)*100

print(Tave,error,poly_order)

#T_string += "\n" + str(round(error,2)) + " %"
#ax[it][0].text(850,np.average(I_calc)/100,T_string)
#
## Emissivity
#ax[it][1].plot(wl_vec,f_eps(wl_vec,Tave),'--')
#ax[it][1].plot(wl_sub_vec,eps_vec_reconstructed) 
#ax[it][1].plot(wl_sub_vec,eps_vec,'-.')
#
#it += 1
