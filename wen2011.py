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

from scipy.interpolate import splrep

# Black and gray body
bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
gr_eps = lambda wl,T: 0.1 * np.ones(len(wl))


### Generate some data
data = np.genfromtxt('data/wen-2011/AL5083-radiance.csv', delimiter=',',skip_header=1)
T = 600 # K

noisy_data = data[:,1] / (1e3 * 1e4)
wl_vec = data[:,0] * 1000 # Wavelengths are in micro-meter
pix_vec = np.linspace(0,len(wl_vec)-1,len(wl_vec))
pix_vec = np.array(pix_vec,dtype=np.int64)

# Take the log of the data
log_noisy = np.log(noisy_data)

# Remove the peaks
nopeak = np.copy(log_noisy)

    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Moving average filter
wl = 11
log_med = moving_average(nopeak,wl)

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

#return I_calc,noisy_data,log_med,data_spl,pix_vec_sub

pix_sub_vec = np.copy(pix_vec_sub)
filtered_data = np.copy(log_med)

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
fig, ax = plt.subplots(2,1)
ax[0].semilogy(wl_vec,noisy_data)
ax[0].semilogy(wl_sub_vec,reconstructed_data)
ax[0].set_title("Radiance")

data = np.genfromtxt('data/wen-2011/AL5083-emissivity.csv', delimiter=',',skip_header=1)
eps_xp = data[:,1]
wl_eps_xp = data[:,0] * 1000 # Wavelengths are in micro-meter

ax[1].plot(wl_eps_xp,eps_xp)
ax[1].plot(wl_sub_vec,eps_vec_reconstructed)
ax[1].set_title("Emissivity")

error = np.abs((Tave-T)/T)*100

print(Tave,error)
