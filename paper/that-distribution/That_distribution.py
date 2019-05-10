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

'''
This file contains the necessary code to generate the temperature distributions
as shown in our 2019 paper.
The graphs show the effect of the test emissivity on the distribution of the
two-color predictions of temperature. 
We assume a correct emissivity of 0.5.
'''

### Impots
# Numpy, matplotlib, scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial2
from scipy.stats import skew,kurtosis
from scipy.stats import cauchy
import itertools
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

# Local
import spectropyrometer_constants as sc
import generate_spectrum as gs

import pixel_operations as po

from statistics import tukey_fence

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
    
    return eps * sc.C1 / wl**5 * np.exp(-sc.C2/(T*wl))


def generate_That_distributions(sigma_I, T0,
                                   wl_vec,
                                   nwl,
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

    ### Index of the vectors
    idx = np.arange(0,nwl,1)
    idx = np.array(idx,dtype=np.int64)  
    
    ### Generate combinations
    cmb_pix = []

    for i,j in itertools.combinations(idx,2):
        cmb_pix.append([i,j])#            
    cmb_pix = np.array(cmb_pix)
    
    ### Which wavelengths are associated with the pixel combinations?
    wl_v0 = wl_vec[cmb_pix[:,0]]
    wl_v1 = wl_vec[cmb_pix[:,1]] 

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
    
    
    ### For each test emissivity, create a distribution of Thats
    distall = np.zeros((data['Nthat'],neps+1))    
    for idx in range(neps+1):
        f_eps = lambda wl,T: f_eps_test(idx,wl,T)
        eps0 = f_eps(wl_v0,1)
        eps1 = f_eps(wl_v1,1)
    
        invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
        That = 1/invT
        That *= sc.C2 * ( 1/wl_v1 - 1/wl_v0)
          
        # Filter out some crazy values
#        Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout, method = 'dispersion')
        
        ### Average of all Thats is the estimate of the true temperature
        Tave = np.average(That)
#        print(idx,Tave)
    
        ### Distribution of Thats
        dThat_dist = (That - Tave)/Tave
        distall[:,idx] = np.copy(dThat_dist)
        data['Tbar'][idx] = Tave
    
    data['distall'] = distall
#    ### Write data to disk
#    root = 'variance_calculations/'
#    np.save(root+'cmb_pix.npy',cmb_pix)
#    np.save(root+'I_calc.npy',I_calc)
#    np.save(root+'wl_v1.npy',wl_v1)
#    np.save(root+'wl_v0.npy',wl_v0)
#    np.save(root+'eps0.npy',eps0)
#    np.save(root+'eps1.npy',eps1)
#    np.save(root+'filtered_data.npy',filtered_data)
        
    return data

### Define our core case
# Noise
sigma_I = 0.1

# Temperature
T0 = 3000 # K

# Wavelengths
wlrange = 1.46
lambda1 = 300 # nm
lambdaN = (1+wlrange)*lambda1
nwl = 50
wl_vec = np.linspace(lambda1,lambdaN,nwl)

# True emissivity
f_eps_true = lambda wl,T: 0.5 * np.ones(len(wl))
   
# Test emissivity
neps = 3
t1 = np.linspace(0.1,1,neps)
tn = t1[::-1]

eps1vec = np.zeros(neps+1)
epsnvec = np.zeros(neps+1)
eps1vec[:-1] = t1
eps1vec[-1] = 0.5

epsnvec[:-1] = tn
epsnvec[-1] = 0.5


mvec = 1/(lambdaN-lambda1) * (epsnvec-eps1vec)
bvec = lambdaN * eps1vec - lambda1 * epsnvec
bvec /= (lambdaN-lambda1)

f_eps_test = lambda idx,wl,T: mvec[idx] * wl + bvec[idx]

### Generate the That distributions
data = generate_That_distributions(sigma_I,T0,wl_vec,nwl,f_eps_true,f_eps_test,neps)

etavec = np.zeros(neps+1)

### Best T prediction
err = 100 * (data['Tbar']/T0-1)
err = np.abs(err)
bidx = np.argmin(err)

print(bidx, err[bidx], data['Tbar'][bidx])

for idx,dist in enumerate(data['distall'].T):
    f_eps = lambda wl,T: f_eps_test(idx,wl,T)
    Tave = data['Tbar'][idx]
    
    ### Remove extreme values
    dmin = -0.1
    dmax = 0.1
    dist_filt = dist[(dist<dmax) & (dist>dmin)]
    dist_filt *= Tave
    dist_filt += Tave
    
    dmin = 1000
    dmax = 5000
    
    if len(dist_filt) < 2 or Tave < 0:
        etavec[idx] = 1e5
        continue
    
#    dist_filt = dist
#    dmin = np.min(dist_filt)
#    dmax = np.max(dist_filt)
    
    ### Find the best bandwidth for KDE
    bandwidths = 10 ** np.linspace(1,2, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=5,
                        verbose = 1,
                        n_jobs = -2)
    grid.fit(dist_filt[:, None])
    
    
    
    ### KDE representation
    kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], 
                        kernel='gaussian')
    kde.fit(dist_filt[:, None])
    x_d = np.linspace(dmin,dmax,1000)
#    
    logprob_kde = kde.score_samples(x_d[:, None])
    pdfkde = np.exp(logprob_kde)
    
    # Location of the maximum of the PDF
    xmax = x_d[np.argmax(pdfkde)]
    
#    plt.plot(x_d,pdfkde)
    
    ### Fit a Cauchy distribution and calculate distance
    loc,scale = cauchy.fit(dist_filt)
    ncauchy = cauchy.pdf(x_d,loc=loc,scale=scale)
#    plt.plot(x_d,cauchyval,'--')
    
#    ### Compute muD, sigmaD, Teq
#    sigD = np.sqrt(2)*sigma_I
#    Tave = data['Tbar'][idx]
#    wl_v1 = data['wl_v1']
#    wl_v0 = data['wl_v0']
#    ncomb = len(wl_v1)
#    
#    eps0 = f_eps(wl_v0,1)
#    eps1 = f_eps(wl_v1,1)
#    Ii = wien_approximation(wl_v0,Tave,f_eps)
#    Ij = wien_approximation(wl_v1,Tave,f_eps)
#    
#    muD = np.log(Ii) - np.log(Ij) - 5*np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
#    Teq = sc.C2 * (1/wl_v1-1/wl_v0)
#    Teq /= muD
#    
#    ### Compute muThat, sigThat
#    muThat = Teq / Tave
#    sigThat = np.abs(Teq / Tave * sigD / muD)
#        
#    ### Setup the mixture distribution
#    pdf_mixture = lambda x: np.sum(1/ncomb * 1/sigThat * 1/np.sqrt(2*np.pi) * np.exp(-1/2*(x-(muThat-1))**2/sigThat**2))
#    
#    ### Setup the Cauchy distribution
#    gamma = ncomb * np.sqrt(2/np.pi) * 1/np.sum(1/sigThat)
#    pdf_cauchy = lambda x: 1/(np.pi*gamma) * gamma**2 / ((x-xmax)**2 + gamma**2)
#    
#    ### Numerically evaluate the distributions
#    ncauchy = pdf_cauchy(x_d)
#    nmixture = [pdf_mixture(x) for x in x_d]
#    nmixture = np.array(nmixture)
    
    
    boolidx = pdfkde > 1e-8
    eta = np.sum((ncauchy[boolidx]/pdfkde[boolidx] - 1 )**2)
    
    etavec[idx] = eta
    
#    p = plt.plot(x_d,pdfkde)
#    plt.plot(x_d,ncauchy,linestyle='dashed',color=p[-1].get_color())
    
    print(idx,dmin,dmax,np.abs(np.mean(dist)),grid.best_params_['bandwidth'],eta)
#    print(idx,dmin,dmax,np.abs(np.mean(dist)),eta)
    
#    plt.hist(dist,bins=100,normed=True,histtype='step')