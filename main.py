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

from scipy.interpolate import splrep

from generate_spectrum import wien_approximation
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

# Intensity with addeded noise
I_calc = wien_approx(lnm_vec,T,chosen_eps)
noisy_data = np.random.normal(I_calc,0.1*I_calc)


window_length = (int)(pix_slice/3)
if window_length % 2 == 0:
    window_length += 1

#filtered_data = savgol_filter(noisy_data,pix_slice+1,5,mode='nearest')
#filtered_data = moving_average(noisy_data,window_length)

log_noisy = np.log10(noisy_data)

window_length = (int)(pix_slice/3)
# Edge mode is acceptable for higher values of lambda
data_padded = np.pad(log_noisy, (window_length//2, window_length-1-window_length//2), mode='edge')

# Not so much for lower values; a linear fit to the data is better there
m_dp,b_dp = np.polyfit(lnm_vec[window_length:2*window_length],
                 data_padded[window_length:2*window_length],1)

#data_padded[0:window_length] = m_dp*lnm_vec[0:window_length] + b_dp
filtered_data = np.convolve(data_padded, np.ones((window_length,))/window_length, mode='valid')

### Remove the edge effects
#lnm_vec = lnm_vec[window_length:-window_length]
lnm_vec_sub = lnm_vec[window_length:-window_length]
filtered_data = filtered_data[window_length:-window_length]
pix_vec = pix_vec[window_length:-window_length]


### Fit a line through the noise with some smoothing
#spl = splrep(lnm_vec,np.log10(filtered_data))
spl = splrep(lnm_vec_sub,filtered_data)





### Plots            
bb_reconstructed = wien_approx(lnm_vec_sub,Tave,bb_eps)
eps_vec = 10**filtered_data/bb_reconstructed
# Since we get epsilon from the filtered data, "reconstructed_data" will be
# exactly like "filtered_data"
reconstructed_data = bb_reconstructed * eps_vec # exactly filtered
    
## Subplots
f, (ax1, ax2) = plt.subplots(1, 2)
# Plot the intensity
ax1.semilogy(lnm_vec,noisy_data)
ax1.semilogy(lnm_vec_sub,reconstructed_data)
#ax1.semilogy(lnm_vec,)

# Plot the emissivity
ax2.plot(lnm_vec_sub,eps_vec)
ax2.plot(lnm_vec,chosen_eps(lnm_vec,Tave),'--')

if refined_fit:
    eps_poly = np.polynomial.Chebyshev(sol.x,[np.min(lnm_vec),np.max(lnm_vec)])
    eps_val = np.polynomial.chebyshev.chebval(lnm_vec,eps_poly.coef)
    
    ax2.plot(lnm_vec,eps_val,'-.')

#epsret = eps_piecewise(sol.x,lnm_vec,lnm_binm,lnm_binM)
#ax2.plot(lnm_vec,epsret,'-.')

