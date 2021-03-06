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

from scipy.interpolate import splrep
from scipy.signal import medfilt, find_peaks_cwt

import algorithm.spectropyrometer_constants as sc

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def wien_approximation(wl,T,f_eps):    
    '''
    Function: wien_approximation
    Calculates the Wien approximation to Planck's law for non-constant emissivity
    Inputs:
        - lnm: wavelength in nm
        - T: temperature in K
        - f_eps: a lambda function representing the emissivity as function of
        temperature and wavelength
    '''
    eps = f_eps(wl,T) # Emissivity
    
    return eps * sc.C1 / wl**5 * np.exp(-sc.C2/(T*wl))

def generate_data(wl_vec,T,pix_vec,f_eps,el = None):
    '''
    Function: generate_data
    Computes an artificial spectrum with noise
    Inputs:
        - wl_vec: vector of wavelengths
        - T: the target temperature
        - pix_vec: the vector of pixel indices
        - f_eps: the emissivity chosen
        - el: emission lines
    Ouputs:
        - I_calc: the Wien approximation of the spectrum w/o any noise
        - noisy_data: I_calc but with some noise
        - filtered_data: the filtered 10-base logarithm of noisy_data
        - data_spl: spline representation of the filtered data
        - pix_vec_sub: the subset of pixels that we are dealing with (we remove 
        the edges after the moving average)
    '''
    # Intensity from Wien's approximation: true data
    I_calc = wien_approximation(wl_vec,T,f_eps)
    
    # Add some emission lines
    if el is not None:
        el_out = generate_emission_line(el, wl_vec, I_calc)
        I_calc += el_out
    
    # Add some noise and take the log of the data
    noisy_data = np.random.normal(I_calc,0.1*I_calc)
    log_noisy = np.log(noisy_data)
    
    # Find the peaks in the data from emission lines
    # This is currently disabled, but it can be un-commented
#    peaks = find_peaks_cwt(log_noisy,np.array([sc.window_length/2]))
#    
#    # Remove the peaks
    nopeak = np.copy(log_noisy)
#    
#    for peak in peaks:
#        # Create a "window" around the peak
#        pxm = (int)(peak-sc.window_length/2)
#        pxM = (int)(peak+sc.window_length/2)
#        
#        # Find the average data before and after the window
#        win_mm = (int)(pxm - sc.window_length)
#        win_mp = (int)(pxm)
#        win_Mm = (int)(pxM)
#        win_Mp = (int)(pxM + sc.window_length)
#        
#        ym = np.average(log_noisy[win_mm:win_mp])
#        yM = np.average(log_noisy[win_Mm:win_Mp])
#        
#        # Fit a polynomial from one end to the other one
#        fit = np.polyfit(np.array([pxm,pxM]),np.array([ym,yM]),deg=1)
#        
#        # Overwrite
#        nopeak[pxm:pxM+1] = np.arange(pxm,pxM+1,1) * fit[0] + fit[1]
    
    # Moving average filter
    wl = sc.window_length
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
    
    return I_calc,noisy_data,log_med,data_spl,pix_vec_sub
    
def generate_emission_line(wl_line, wl_vec, I_calc, fac = 10):
    '''
    Function: generate_emission_line
    Creates an emission line at a given wavelength which is 50% larger in
    intensity than the base intensityt
    Inputs:
        - wl_line: line wavelength in nm
        - wl_vec: all wavelengths in nm
        - I_calc: the base intensity vector
        - fac: a growth factor. The line will be fac * I_base at its peak
    
    '''
    # Number of pixels
    length = len(wl_vec)
    
    v_out = np.zeros(length)
    
    broadening = np.zeros(length)
    sig = 2 # 10 nm standard deviation
    for wl in wl_line:
        cond = np.abs(wl_vec-wl) < 0.15
        br = 1/(np.sqrt(np.pi)*sig)*np.exp(-1/2*((wl_vec-wl)/sig)**2)
        # Rescale Intensity value
        Ival = I_calc[cond]
        br *= fac * Ival
        
        broadening += br
    
    v_out += broadening
    
    return v_out

