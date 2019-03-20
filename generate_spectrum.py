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
from scipy.signal import medfilt

import spectropyrometer_constants as sc

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


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
    # Intensity from Wien's approximation
    I_calc = wien_approximation(wl_vec,T,f_eps)
    
    if el is not None:
        el_out = generate_emission_line(el, wl_vec, I_calc)
        I_calc += el_out
    
    # Add some noisy and take the log base 10
    noisy_data = np.random.normal(I_calc,0.1*I_calc)
    log_noisy = np.log10(noisy_data)
    
    # 
    log_smooth = smooth(log_noisy,window_len=sc.medfilt_kernel)
    wl_smooth = smooth(wl_vec,window_len=sc.medfilt_kernel)
    print(len(log_smooth),len(pix_vec))
    
    # Median filter
    log_med = medfilt(log_smooth,sc.window_length+1)
    
    # Moving average
    wl = sc.window_length
#    data_padded = np.pad(log_med, 
#                         (wl//2, wl-1-wl//2), 
#                         mode='edge')
    
#    # Not so much for lower values; a linear fit to the data is better there
#    m_dp,b_dp = np.polyfit(wl_vec[window_length:2*window_length],
#                     data_padded[window_length:2*window_length],1)
    
    #data_padded[0:window_length] = m_dp*lnm_vec[0:window_length] + b_dp
#    filtered_data = np.convolve(data_padded, 
#                                np.ones((window_length,))/window_length, 
#                                mode='valid')

#    filtered_data = np.convolve(data_padded, 
#                                np.ones((wl,))/wl, 
#                                mode='valid')
    
#    filtered_data = smooth(log_med,window_len=1)
    
    
    ### Remove the edge effects
    wl_vec_sub = wl_vec[wl:-1]
    log_med = log_med[wl:-wl]
    pix_vec_sub = pix_vec[wl:-1]
    
    print(len(log_med),len(wl_vec[wl:-1]))
    
    ### Fit a line through the noise with some smoothing
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
    
    # Conditional
#    cond = np.zeros(length,dtype=np.bool)
#    for wl in wl_line:
#        cond |= np.abs(wl_vec-wl) < 0.15
    
#    I_base = np.zeros(length)
#    I_base = np.copy(I_calc)
    
    v_out = np.zeros(length)
    
#    v_out = fac * I_base * cond
    
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
    
    
    

    
    
    