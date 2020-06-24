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
File: error-vs-dispersion.py

Description: This file contains the necessary code to generate the graphs 
that shows the effect of the test emissivity on average error and coefficient
of dispersion. We assume a correct emissivity of 0.5.

The corresponding figure is Fig. 6 in our 2020 RSI Journal article.
'''

### Imports
# Numpy, matplotlib, scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import iqr

import itertools


C1 = 1.191e16 # W/nm4/cm2
C2 = 1.4384e7 # nm K

### Function definitions
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

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


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

def generate_That_distributions(sigma_I, T0,
                                   wl_vec,
                                   nwl,
                                   wdw,
                                   wdwo2,
                                   f_eps_true,
                                   f_eps_test,
                                   neps):
    '''
    This function generates a distribution of errors for a single measurement. 
    The errors are between the true temperature and the one calculated with 
    the variance method.
    Inputs:
        - T0: true temperature
        - wl_vec: the vector of wavelengths
        - pix_vec: the vector of pixels
        - nwl: the number of wavelengths to consider
        - nthat: number of Monte-Carlo sample to generate
    
    '''
    ### Dictionary to store data
    data = {}
    
    ### Sample true data    
    # Intensity from Wien's approximation
    I_calc = wien_approximation(wl_vec,T0,f_eps_true)
    
    # Add some noise and take the log base 10
    noisy_data = np.random.normal(I_calc,sigma_I*I_calc)
    log_noisy = np.log(noisy_data)
    
    # Filter
#    log_noisy = moving_average(log_noisy,wdw)
    if wdwo2 > 0:
        log_noisy = log_noisy[wdwo2:-wdwo2]
        
        # Rearrange the indices
        lwl_vec = wl_vec[wdwo2:-wdwo2]
    else:
        lwl_vec = np.copy(wl_vec)

    
    ### Index of the vectors
    idx = np.arange(0,nwl,1)
    idx = np.array(idx,dtype=np.int64)  
    
    ### Generate combinations
    cmb_pix = []

    for i,j in itertools.combinations(idx,2):
        cmb_pix.append([i,j])           
    cmb_pix = np.array(cmb_pix)
    
    ### Which wavelengths are associated with the pixel combinations?
    wl_v0 = lwl_vec[cmb_pix[:,0]]
    wl_v1 = lwl_vec[cmb_pix[:,1]] 

    ### Calculate intensity ratio
    logIi = log_noisy[cmb_pix[:,0]]
    logIj = log_noisy[cmb_pix[:,1]]

    logR = (logIi-logIj)
    
    ### Store the ground truth
    data['Icalc'] = np.copy(I_calc)
    data['noisy_data'] = np.copy(noisy_data)
    data['log_noisy'] = np.copy(log_noisy)
    data['wl_v0'] = np.copy(wl_v0)
    data['wl_v1'] = np.copy(wl_v1)
    
    R = np.max(wl_v1)/np.min(wl_v0) - 1
    data['R'] = R
    data['lambda1'] = np.min(wl_v0)
    data['lambdaN'] = np.max(wl_v1)
    data['Nwl'] = nwl
    data['Nthat'] = (int)(nwl*(nwl-1)/2)
    data['T0'] = T0
    data['true_eps'] = f_eps_true
    data['Tbar'] = np.zeros(neps+1)
    data['metric'] = np.zeros(neps+1)
    
    
    ### For each test emissivity, create a distribution of Thats
    distall = np.zeros((data['Nthat'],neps+1))    
    for idx in range(neps+1):
        f_eps = lambda wl,T: f_eps_test(idx,wl,T)
        eps0 = f_eps(wl_v0,1)
        eps1 = f_eps(wl_v1,1)
    
        invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
        That = 1/invT
        That *= C2 * ( 1/wl_v1 - 1/wl_v0)
          
        # Filter out outliers
        Tave, Tstd, Tmetric, Tleft = tukey_fence(That)
        
        ### Average of all Thats is the estimate of the true temperature
        Tave = np.average(Tleft)
    
        ### Distribution of Thats
        dThat_dist = (That - Tave)/Tave
        distall[:,idx] = np.copy(dThat_dist)
        data['Tbar'][idx] = Tave
        data['metric'][idx] = Tmetric
    
    data['distall'] = distall

        
    return data

### Define our core case
# Controls
test_emissivity_order = 1

# Noise
sigma_I = 0.1

# Temperature
T0 = 3000 # K

# Wavelengths
wlrange = 1.46 # wavelength range
lambda1 = 300 # nm
lambdaN = (1+wlrange)*lambda1 # nm
nwl = 50 # number of wavelengths
wdw = 21 # window size
wdwo2 = (int)((wdw-1)/2)

wl_vec = np.linspace(lambda1,lambdaN,nwl)

dlambda = np.abs(wl_vec[0]-wl_vec[1])

# Add window for moving average
wl_vec = np.linspace(lambda1 - wdwo2 * dlambda, 
                     lambdaN + wdwo2 * dlambda, 
                     nwl + wdw - 1)


# True emissivity
f_eps_true = lambda wl,T: 0.5 * np.ones(len(wl))
   
# Test emissivities
neps = 500
epsm = 0.01 # Minimum emissivty
epsM = 1 # Maximum emissivity
    
if test_emissivity_order == 1:  
    t1 = np.linspace(epsm,epsM,neps)
    tn = t1[::-1]
    eps1vec = np.zeros(neps+1)
    epsnvec = np.zeros(neps+1)
    eps1vec[:-1] = t1
    eps1vec[-1] = 0.5
    
    epsnvec[:-1] = tn
    epsnvec[-1] = 0.5
    
    eps1vec = np.zeros(neps+1)
    epsnvec = np.zeros(neps+1)
    eps1vec[:-1] = t1
    eps1vec[-1] = 0.5
    
    epsnvec[:-1] = tn
    epsnvec[-1] = 0.5
    
    lambdam = np.min(wl_vec)
    lambdaM = np.max(wl_vec)
    
    
    mvec = 1/(lambdaM-lambdam) * (epsnvec-eps1vec)
    bvec = lambdaM * eps1vec - lambdam * epsnvec
    bvec /= (lambdaM-lambdam)
    
    f_eps_test = lambda idx,wl,T: mvec[idx] * wl + bvec[idx]

elif test_emissivity_order == 2:
## Order 2 test functions
    eps1 = np.linspace(epsM,epsm,neps)
    epsN = np.linspace(epsM,epsm,neps)
    epsmid = np.linspace(epsm,epsM,neps)
    avec = np.zeros(neps+1)
    bvec = np.zeros(neps+1)
    cvec = np.zeros(neps+1)
    
    lvec = np.array([lambdam,(lambdaM+lambdam)/2,lambdaM])
    for idx in range(neps):
        epsv = np.array([eps1[idx],epsmid[idx],epsN[idx]])
        p = np.polyfit(lvec,epsv,2)
        
        avec[idx] = p[0]
        bvec[idx] = p[1]
        cvec[idx] = p[2]
    
    avec[-1] = 0
    bvec[-1] = 0
    cvec[-1] = 0.5
        
    f_eps_test = lambda idx,wl,T: avec[idx] * wl**2 + bvec[idx] * wl + cvec[idx]


### Generate data that many times
ntest = 1000
errall = np.zeros((neps+1,ntest))
metricall = np.zeros((neps+1,ntest))

for testidx in range(ntest):
    ### Generate the That distributions
    data = generate_That_distributions(sigma_I,T0,wl_vec,nwl,wdw,wdwo2,
                                       f_eps_true,f_eps_test,neps)
    
    metricvec = np.zeros(neps+1)
    
    ### Best T prediction
    err = 100 * (data['Tbar']/T0-1)
    err = np.abs(err)
    bidx = np.argmin(err)
    
    errall[:,testidx] = np.copy(err)
    
    if np.mod(testidx,100) == 0:
        print(testidx)
    
    for idx,dist in enumerate(data['distall'].T):
        f_eps = lambda wl,T: f_eps_test(idx,wl,T)
        Tave = data['Tbar'][idx]
        
        dist_filt = np.copy(dist)
        dist_filt *= Tave
        dist_filt += Tave
        
        Tave, Tstd, metric, Tleft = tukey_fence(dist_filt)
        
        dist_filt = np.copy(Tleft)
        dmin = np.min(dist_filt)
        dmax = np.max(dist_filt)
        
        if len(dist_filt) < 2 or Tave < 0:
            metricvec[idx] = 1e10
            continue
                
        metricvec[idx] = data['metric'][idx]
        

    metricall[:,testidx] = np.copy(metricvec)

errplt = np.nanmean(errall,axis=1)
metplt = np.nanmean(metricall,axis=1)

boolidx = (metplt > 0) & (errplt < 15)
metplt_filtered = metplt[boolidx]
errplt_filtered = errplt[boolidx]

#plt.figure()
#plt.plot(np.mean(varall,axis=1),errplt,'k.')
plt.plot(metplt_filtered[:-1],errplt_filtered[:-1])

plt.plot(metplt[-1],errplt[-1],'k*',markersize=12)
plt.xlabel('Dispersion')
plt.ylabel('Error from true temperature (%)')
