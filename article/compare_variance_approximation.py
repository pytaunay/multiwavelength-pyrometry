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

'''
File: compare_variance_approximations

Desription: Calculate the variance and standard error with or without an 
asymptotic expansion.

This file generates Fig. 4 in our 2020 RSI Journal article.
'''

import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, iqr

C1 = 1.191e16 # W/nm4/cm2 Sr
C2 = 1.4384e7 # nm K

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
    
    return eps * C1 / wl**5 * np.exp(-C2/(T*wl))

def tukey_fence(Tvec, delta=31.3):
    '''
    Function: tukey_fence
    Descritpion: Removes outliers using Tukey fencing
    Inputs:
        - Tvec: some vector
        - delta: a fencing value above/below the third/first quartile, 
        respectively. Values outside of [Q1 - delta * IQR, Q3 + delta*IQR] are
        discarded
    Outputs:
        - Average of vector w/o outliers
        - Standard deviation of vector w/o outliers
        - Standard error of vector w/o outliers (%)
        - Vector w/o outliers
    '''      
    ### Exclude data w/ Tukey fencing
    T_iqr = iqr(Tvec)
    T_qua = np.percentile(Tvec,[25,75])
    
    min_T = T_qua[0] - delta * T_iqr
    max_T = T_qua[1] + delta * T_iqr
    
    T_left = Tvec[(Tvec>min_T) & (Tvec<max_T)]
    
    ### Calculate standard deviation, average of the fenced data
    Tstd = np.std(T_left)
    Tave = np.mean(T_left)
    
    ### Calculate a metric: coefficient of quartile dispersion
    dispersion = (T_qua[1] - T_qua[0]) / (T_qua[1] + T_qua[0])
    metric = dispersion

    return Tave, Tstd, metric, T_left

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def generate_Taverage_distribution(sigma_I, 
                                   T0,
                                   wl_vec,
                                   nwl,
                                   wdw,
                                   wdwo2,
                                   ntbar):
    '''
    This function generates a distribution of errors. The errors are between
    the true temperature and the one calculated with the variance method.
    Inputs:
        - T0: true temperature
        - wl_vec: the vector of wavelengths
        - pix_vec: the vector of pixels
        - nwl: the number of wavelengths to consider
        - ntbar: number of Monte-Carlo sample to generate
    
    '''
    ### Grey body
    gr_eps = lambda wl,T: 0.5 * np.ones(len(wl))
    
    # Distribution of Taverage over T
    Tbar_dist = np.zeros(ntbar)

    ### Dictionary to store data
    data = {}
    
    ### Sample true data    
    # Intensity from Wien's approximation
    I_calc = wien_approximation(wl_vec,T0,gr_eps)
        
    ### Sample data
    for idx in range(ntbar):
        # Add some noise and take the natural log
        noisy_data = np.random.normal(I_calc,sigma_I*I_calc)
        log_noisy = np.log(noisy_data)
        
        # Filter
        if wdwo2 > 0:
            log_noisy = moving_average(log_noisy,wdw)
            
            # Rearrange the indices
            lwl_vec = wl_vec[wdwo2:-wdwo2]
        else:
            lwl_vec = np.copy(wl_vec)
        
        ### Index of the vectors
        vidx = np.arange(0,nwl,1)
        vidx = np.array(vidx,dtype=np.int64)  
            
        ### Generate combinations
        cmb_pix = []
    
        for i,j in itertools.combinations(vidx,2):
            cmb_pix.append([i,j])           
        cmb_pix = np.array(cmb_pix)
        
        ### Which wavelengths are associated with the pixel combinations?
        wl_v0 = lwl_vec[cmb_pix[:,0]]
        wl_v1 = lwl_vec[cmb_pix[:,1]] 
    
        ### Calculate intensity ratio
        logIi = log_noisy[cmb_pix[:,0]]
        logIj = log_noisy[cmb_pix[:,1]]
    
        logR = (logIi-logIj)
        
        # No emissivity error, so we can calculate eps1 and eps0 directly
        # from the given emissivity function
        eps0 = gr_eps(wl_v0,1)
        eps1 = gr_eps(wl_v1,1)
    
        invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
        That = 1/invT
        That *= C2 * ( 1/wl_v1 - 1/wl_v0)
          
        # Filter out outliers
        Tave, Tstd, Tmetric, Tleft = tukey_fence(That)
        
        ### Average of all Thats is the estimate of the true temperature
        Tave = np.average(Tleft)
                    
        ### Distributions
        Tbar_dist[idx] = (Tave - T0)/T0
                
    data['Icalc'] = np.copy(I_calc)
    data['noisy_data'] = np.copy(noisy_data)
    data['log_noisy'] = np.copy(log_noisy)
    data['cmb_pix'] = np.copy(cmb_pix)
    data['wl_v0'] = np.copy(wl_v0)
    data['wl_v1'] = np.copy(wl_v1)

    R = np.max(wl_v1)/np.min(wl_v0) - 1
    data['R'] = R
    data['lambda1'] = np.min(wl_v0)
    data['lambdaN'] = np.max(wl_v1)
    data['Nwl'] = nwl
    data['Nthat'] = (int)(nwl*(nwl-1)/2)
    data['T0'] = T0
    data['Tbar'] = np.copy(Tbar_dist)
    data['That'] = np.copy(That)
    
    return data


def compute_high_order_variance(sigma_I,T0,nwl,wdw,wdwo2,data):  
    '''
    Calculates the variance from the successive Taylor expansions. We keep
    high orders for all of the expansions.
    Inputs:
        - T0: true temperature
        - sigma_I: standard variation on the measurement noise
    '''
    wl_v1 = data['wl_v1']  
    wl_v0 = data['wl_v0']
    I_calc = data['Icalc']
    cmb_pix = data['cmb_pix']
    
    # Number of combinations
    ncomb = len(wl_v1)
    
    ### Denominator average and variance for all wavelengths
    # mud, sigd
    mud = np.zeros(ncomb)
    logI = np.log(I_calc)
    
    # Filtering effect on mu_d
    logI = moving_average(logI,wdw)
    
    mud = logI[cmb_pix[:,0]] - logI[cmb_pix[:,1]]
    
    # No epsilon here: we assumed constant emissivity
    mud += - 5*np.log(wl_v1/wl_v0)
    
    sigd = np.ones_like(mud) * np.sqrt(2/wdw)*sigma_I
    mudmin = np.min(np.abs(mud))
    
    # sigma_d / mu_d
    ratio = np.unique(sigd) / mudmin
    ratio = np.abs(ratio)
    
    ### muThat, sigThat
    muThat = np.zeros_like(mud)
    sigThat = np.zeros_like(mud)
    
    # Teq
    Teq = C2*(1/wl_v1-1/wl_v0) / mud
    
    # Taylor expansion for muThat: we want to find the "best" order in the 
    # expansion of the mean of 1/X
    sigomud = sigd/mud
    sigomud2 = sigomud**2
    
    muThat = Teq/T0 
    
    # Taylor expansion for sigThat: we keep only the first two orders
    sigThat = (Teq/T0)**2 * sigomud2 * (1 + 2*sigomud2)
    sigThat = np.sqrt(sigThat)
        
    ### muTbar, sigTbar
    # muTbar: subtract 1 bc. we want (Tbar-T0)/T0 = Tbar/T0 - 1
    muTbar = 1/ncomb * np.sum(muThat) - 1
    
    # sigTbar
    sigTbar = 1/ncomb**2 * np.sum(sigThat**2)
    sigTbar = np.sqrt(sigTbar)
    
    
    return muTbar,sigTbar,ratio,muThat,sigThat


### Input parameters
ntbar = 1000 # Number of samples for Monte-Carlo
T0 = 1500
sigma_I = 0.01
wdw = 1 # window size
wdwo2 = (int)((wdw-1)/2)

# Wavelengths
wlRange = 1.46

# Holder for results
res = []

for lambda1 in np.array([300]):
    lambdaN = (1+wlRange) * lambda1
    
    ### Approximate the starting point   
    sigd = np.sqrt(2/wdw) * sigma_I
    rlim = 0.1
    Napprox = 1
    Napprox += C2 / (T0*lambda1) * wlRange / (1+wlRange)**2 * rlim / sigd
    
    nwl_array = np.arange(10,100,10)
    nwl_array = np.array(nwl_array,dtype=np.int64)
    print("Napprox = ", Napprox)

    for idx,nwl in enumerate(nwl_array):
        
        if np.mod(idx,10) == 0:
            print(idx)
        
        wl_vec = np.linspace(lambda1,lambdaN,nwl)        
        
        dlambda = np.abs(wl_vec[0]-wl_vec[1])
        
        # Add window for moving average
        wl_vec = np.linspace(lambda1 - wdwo2 * dlambda, 
                             lambdaN + wdwo2 * dlambda, 
                             nwl + wdw - 1)
        
        ### Create some data
        data = generate_Taverage_distribution(sigma_I, 
                                   T0,
                                   wl_vec,
                                   nwl,
                                   wdw,
                                   wdwo2,
                                   ntbar)
        muds,sigds = norm.fit(data['Tbar'])
        
        ### Calculate the variance based on the second-order accurate expansions
        muAcc, sigAcc, ratio, muThat, sigThat = compute_high_order_variance(sigma_I,T0,nwl,wdw,wdwo2,data)
        
        res.append([C2/(T0*lambda1), # 0
                    nwl, # 1 
                    dlambda, # 2
                    ratio, # 3 
                    sigds, # 4
                    sigAcc, # 5
                    muds * 100, # 6
                    muAcc * 100]) # 7

res = np.array(res)

fig, ax = plt.subplots(2,1)

ax[0].vlines(Napprox,np.min(res[:,5]),np.max(res[:,5]),linestyles='--')
ax[0].plot(res[:,1],res[:,4],'^')
ax[0].plot(res[:,1],res[:,5])
ax[0].set_ylabel("Standard deviation")

ax[1].plot(res[:,1],res[:,6],'^')
ax[1].plot(res[:,1],res[:,7])
ax[1].set_ylabel("Error of the mean (%)")