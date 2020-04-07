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
from numpy.polynomial import Polynomial, polynomial

import matplotlib.pyplot as plt
import generate_spectrum as gs

from pixel_operations import choose_pixels, generate_combinations
from temperature_functions import optimum_temperature
from kfold import order_selection

from scipy.interpolate import splrep,splev

### Emissivity functions
# Black and gray body
bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
gr_eps = lambda wl,T: 0.1 * np.ones(len(wl))


# Emissivity data
data = np.genfromtxt('data/koirala-2004/al2o3_2360K.csv', delimiter=',',skip_header=1)
T = 2360

# Transform to wavelength (nm) and sort by increasing wavelength
wl_eps_xp = 1/data[:,0] * 1e7
eps_xp = data[:,1]

ret = sorted(zip(wl_eps_xp,eps_xp))
ret = np.array(ret)
wl_eps_xp = ret[:,0]
eps_xp = ret[:,1]

# Spline the data to generate the "true" emissivity
eps_spl = splrep(wl_eps_xp,eps_xp)
f_eps = lambda wl,T: splev(wl,eps_spl)

### Vectors of pixels and wavelengths
wl_vec = np.linspace(np.min(wl_eps_xp),np.max(wl_eps_xp),(int)(3000))
pix_vec = np.linspace(0,2999,3000)
pix_vec = np.array(pix_vec,dtype=np.int64)

### Generate some data
I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
        wl_vec,T,pix_vec,f_eps,None)

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

ax[1].plot(wl_eps_xp,eps_xp)
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
T_string = str(round(Tave,1)) + "+/-" + str(round(Tstd,2)) + " K"
error = np.abs((Tave-T)/T)*100
print(T,Tave,error)

#T_string += "\n" + str(round(error,2)) + " %"
#ax[0].text(3000,np.average(I_calc)/100,T_string)
