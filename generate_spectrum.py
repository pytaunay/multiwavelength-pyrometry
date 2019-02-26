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

from scipy.interpolate import splrep

from spectropyrometer_constants import C1,C2,window_length

'''
Function: wien_approximation
Calculates the Wien approximation to Planck's law for non-constant emissivity
Inputs:
    - lnm: wavelength in nm
    - T: temperature in K
    - f_eps: a lambda function representing the emissivity as function of
    temperature and wavelength
'''
def wien_approximation(wl,T,f_eps):    
    eps = f_eps(wl,T) # Emissivity
    
    return eps * C1 / wl**5 * np.exp(-C2/(T*wl))

'''
Function: generate_artificial_spectrum
Computes an artificial spectrum with noise
Inputs:
    - wl_vec: vector of wavelengths
    - T: the target temperature
    - f_eps: the emissivity chosen
'''
def generate_data(wl_vec,T,f_eps):
    # Intensity from Wien's approximation
    I_calc = wien_approximation(wl_vec,T,f_eps)
    
    # Add some noisy and take the log base 10
    noisy_data = np.random.normal(I_calc,0.1*I_calc)
    log_noisy = np.log10(noisy_data)
    
    # Moving average
    data_padded = np.pad(log_noisy, 
                         (window_length//2, window_length-1-window_length//2), 
                         mode='edge')
    
#    # Not so much for lower values; a linear fit to the data is better there
#    m_dp,b_dp = np.polyfit(wl_vec[window_length:2*window_length],
#                     data_padded[window_length:2*window_length],1)
    
    #data_padded[0:window_length] = m_dp*lnm_vec[0:window_length] + b_dp
    filtered_data = np.convolve(data_padded, 
                                np.ones((window_length,))/window_length, 
                                mode='valid')
    
    ### Remove the edge effects
    wl_vec_sub = wl_vec[window_length:-window_length]
    filtered_data = filtered_data[window_length:-window_length]
    pix_vec = pix_vec[window_length:-window_length]
    
    
    ### Fit a line through the noise with some smoothing
    data_spl = splrep(wl_vec_sub,filtered_data)
    
    
    return I_calc,noisy_data,filtered_data,data_spl
    