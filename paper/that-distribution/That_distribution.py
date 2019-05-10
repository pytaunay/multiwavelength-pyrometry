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

def muthat_expansion(mu,sig,order):
    '''
    Calculates the expansion for mu_{\hat{T}} based on mu_D and sigma_D.
    The expansion calculates the average of 1/X where X ~ N(mu_D,sigma_D^2):
        E[1/X] \approx 1/mu * sum_{k=0}^N (sigma/mu)**2k * (2k-1)!!
    Because 1/mu is lumped into Teq after the call to this function, it is
    not necessary to multiply the result of the for loop by 1//mu.
    Inputs:
        - mu, sig: expected value and standard deviation of X
        - order: order of the expansion

    '''
    sigomu = sig/mu
    sigomu2 = sigomu**2
    approx = 0
    for k in range(order):
        approx += sigomu2**k * factorial2(2*k-1)
        
    return approx


def compute_high_order_variance(T0,sigma_I,w):  
    '''
    Calculates the variance from the successive Taylor expansions. We keep
    high orders for all of the expansions.
    Inputs:
        - T0: true temperature
        - sigma_I: standard variation on the measurement noise
    '''
    root = 'variance_calculations/'
    cmb_pix = np.load(root+'cmb_pix.npy')
    I_calc = np.load(root+'I_calc.npy')
    wl_v1 = np.load(root+'wl_v1.npy')
    wl_v0 = np.load(root+'wl_v0.npy')
    eps0 = np.load(root+'eps0.npy')
    eps1 = np.load(root+'eps1.npy')
    filtered_data = np.load(root+'filtered_data.npy')
    
    # Number of combinations
    Ncomb = len(wl_v1)
    
    ### Denominator average and variance for all wavelengths
    # mud, sigd
    mud = np.zeros(Ncomb)
    for idx in range(Ncomb):
        pixi = cmb_pix[idx,0]
        pixj = cmb_pix[idx,1]
        
        # window size over 2
        wo2 = (int)(sc.window_length/2)
        
#        icontrib = np.sum(np.log(I_calc)[pixi-wo2:pixi+wo2+1])
#        jcontrib = np.sum(np.log(I_calc)[pixj-wo2:pixj+wo2+1])
        icontrib = np.average(np.log(I_calc)[pixi-wo2:pixi+wo2+1])
        jcontrib = np.average(np.log(I_calc)[pixj-wo2:pixj+wo2+1])
        mud[idx] = icontrib - jcontrib
#        mud[idx] = 1/w * (icontrib - jcontrib)
#        mud[idx] = np.log(I_calc)[pixi] -  np.log(I_calc)[pixj] 
#        mud[idx] = np.log(10)*(filtered_data[pixi] - filtered_data[pixj])
    
    mud += - 5*np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
    sigd = np.ones_like(mud) * np.sqrt(2/w)*sigma_I
    mudmin = np.min(np.abs(mud))
    
    
    
    ratio = np.unique(sigd) / mudmin
    ratio = np.abs(ratio)
    
#    print(mudmin,np.unique(sigd),ratio)
    
    ### muThat, sigThat
    muThat = np.zeros_like(mud)
    sigThat = np.zeros_like(mud)
    
    # Teq
    Teq = sc.C2*(1/wl_v1-1/wl_v0) / mud
    
    # Taylor expansion for muThat: we want to find the "best" order in the 
    # expansion of the mean of 1/X
    sigomud = sigd/mud
    sigomud2 = sigomud**2
    
    findN = lambda N,idx: np.abs(1/mud[idx]) * factorial2(2*N-1) * sigomud2[idx]**N
    Narr = np.arange(0,30,1)
    Narr = np.array(Narr,dtype=np.int64)
    
    for idx in range(Ncomb):
        argmin = np.argmin(findN(Narr,idx))
        order = (int)(Narr[argmin])
        
        mu = mud[idx]
        sig = sigd[idx]
#        muThat[idx] = muthat_expansion(mu,sig,order)
        muThat[idx] = 1
    
    muThat *= Teq / T0
    
    # Taylor expansion for sigThat: we keep only the first two orders
    sigThat = (Teq/T0)**2 * sigomud2 * (1 + 2*sigomud2)
    sigThat = np.sqrt(sigThat)    
    
    
    ### muTbar, sigTbar
    # muTbar: subtract 1 bc. we want (Tbar-T0)/T0 = Tbar/T0 - 1
    muTbar = 1/Ncomb * np.sum(muThat) - 1
    
    # sigTbar
    sigTbar = 1/Ncomb**2 * np.sum(sigThat**2)
    sigTbar = np.sqrt(sigTbar)
    
    return muTbar,sigTbar,ratio,muThat,sigThat


### Define our core case
# Noise
sigma_I = 0.05    

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

#NOne
#w_wl = np.array([300,350,400,500,600,700,800,900])
#w_eps_data = np.array([0.474,0.473,0.474,0.462,0.448,0.436,0.419,0.401])
#
#w_m,w_b = np.polyfit(w_wl,w_eps_data,deg=1)
#w_eps = lambda wl,T: w_m*wl + w_b
#
## Black and gray body
#bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
#gr_eps = lambda wl,T: 0.5 * np.ones(len(wl))
#
## Artificial tests
#art_wl = np.array([300,500,1100])
#art_eps_data = np.array([1,0.3,1])
#art_fac = np.polyfit(art_wl,art_eps_data,deg=2)
#
#a0,a1,a2 = art_fac
#art_eps = lambda wl,T: a0*wl**2 + a1*wl + a2
#
#### Vectors of pixels and wavelengths
#wl_vec = np.linspace(300,1100,(int)(3000))
#pix_vec = np.linspace(0,2999,3000)
#pix_vec = np.array(pix_vec,dtype=np.int64)
##f_eps = art_eps
#
#### Input parameters
#nthat = 1000 # Number of samples for Monte-Carlo
##T0vec = np.arange(1000,3000,100)
#T0vec = np.array([3000])
#sigma_I = 0.1
#w = sc.window_length + 1
#
## Wavelengths
##Rarray = np.logspace(-1,1,10)
##lambdavec = np.arange(200,1200,200)
#lambdavec = np.array([300])
#Rarray = np.array([2])
#Narray = np.array([200])
##Narray = np.arange(10,500,100)
#Narray = np.array(Narray,dtype=np.int64)
#NvsR = []
#
#res = []
#for lambda_0 in lambdavec:
#    for T0 in T0vec:
#        for Rapprox in  Rarray:   
#    #        lambda_0 = 300
#            lambda_N = (1+Rapprox) * lambda_0
#            
#            for nwl in Narray:
#                chosen_pix = np.linspace(50,2949,nwl)
#                chosen_pix = np.array(chosen_pix,dtype=np.int64)
#                       
#                #lambda_0 = 300
#                #lambda_N = 1100
#        
#                wl_vec = np.linspace(lambda_0,lambda_N,(int)(3000))
#                
#                pix_vec = np.linspace(0,2999,3000)
#                pix_vec = np.array(pix_vec,dtype=np.int64)
#                
#                ncomb = len(chosen_pix)
#                ncomb = (int)(nwl * (nwl-1)/2)
#                
#                _, dToT = generate_Taverage_distribution(T0,wl_vec,pix_vec,nwl)
#                muTbar,sigTbar,ratio,muThat,sigThat = compute_high_order_variance(T0,sigma_I,w)
##                _ = plt.hist(dToT[(dToT<0.1)&(dToT>-0.1)],bins=100,normed=True,histtype='step')
##                dToT = dToT[(dToT<0.1)&(dToT>-0.1)]
#    
#                loc,sca = cauchy.fit(dToT)
#                x_d = np.linspace(-0.1,0.1,1000)
#                plt.plot(x_d,cauchy.pdf(x_d,loc=loc,scale=sca),'k-')
#    
#                mu = np.average(dToT)
#                sig = np.std(dToT)
#                skw = skew(dToT)
#                krt = kurtosis(dToT,fisher=False)
#                
#                res.append([mu,sig,skw,krt])
#
#res = np.array(res)
##plt.xlim([-0.1,0.1])
#
#
#
#import numpy as np
#import matplotlib.pyplot as plt
#
#from scipy.stats import cauchy, norm, t
#
#from sklearn.neighbors import KernelDensity
#from sklearn.grid_search import GridSearchCV
#
#nwl = 200
#ncomb = (int)(nwl * (nwl-1)/2)
#
#alpha = 0.01
#Rp1 = 1 + alpha * (nwl - 1)
#
#sigarr = []
#sigarr_expon = []
#sigarr_div = []
#ijarr = []
#
#### Create a bunch of standard deviations
#for ii in range(nwl-1):
#    for jj in range(ii+1,nwl):
#        lsig = 1/alpha * 1/(ii-jj)
#        lsig *= (1+alpha*ii)
#        lsig *= (1+alpha*jj)
#        sigarr.append(lsig)
#        ij = jj - 1 -ii + 1/2 * (2*nwl - (ii + 1)) * ii
#        ijarr.append(ij)
#        
#        lsig = Rp1**(ii/(nwl-1)) * Rp1**(jj/(nwl-1))
#        lsig /= (Rp1**(ii/(nwl-1)) - Rp1**(jj/(nwl-1)))
#        sigarr_expon.append(lsig)
#        
#        lsig = (1-ii / (ii-(nwl+1))) * (1-jj / (jj-(nwl+1))) 
#        lsig /= (1-ii / (ii-(nwl+1))) - (1-jj / (jj-(nwl+1))) 
#        sigarr_div.append(lsig)
#        
#sigarr = np.array(sigarr)
#sigarr = np.abs(sigarr)
#sigarr_expon = np.array(sigarr_expon)
#sigarr_expon = np.abs(sigarr_expon)
#sigarr_div = np.array(sigarr_div)
#sigarr_div = np.abs(sigarr_div)
#ijarr = np.array(ijarr)
#
#### Sample the nwl * (nwl-1)/2 normal distributions
#Zarr = np.zeros(ncomb)
#for m in range(ncomb):
#    sigrand = sigarr[m]
##    sigrand = 10
##    sigrand = sigarr_expon[m]
##    sigrand = sigarr_div[m]
##    sigrand = np.random.choice(sigarr_expon)
#    Xrand = norm.rvs(loc=0,scale=sigrand,size=1)
#    Zarr[m] = Xrand
#
#### Fit a Cauchy distribution
#loc,sca = cauchy.fit(Zarr)
#locnorm, scanorm = norm.fit(Zarr)
#dft, loct, scat = t.fit(Zarr)
#
#### Compound distribution
##sigarr[:] = sigrand
##weights = 1/sigarr_expon
##weights = weights / np.sum(weights)
#weights = np.ones_like(sigarr)
#pdf_cmb = lambda x: np.sum(weights * 1/sigarr * 1/np.sqrt(2*np.pi) * np.exp(-1/2*x**2/sigarr**2))
##pdf_cmb  = lambda x: np.sum(weights * 1/sigarr_expon * 1/np.sqrt(2*np.pi) * np.exp(-1/2*x**2/sigarr_expon**2))
##pdf_cmb  = lambda x: np.sum(weights * 1/sigarr_div * 1/np.sqrt(2*np.pi) * np.exp(-1/2*x**2/sigarr_div**2))
#
#### Buhlmann
##v2 = np.var(sigarr)
#
#
#
#### KDE
#print("KDE")
##bandwidths = 10 ** np.linspace(-3, -2, 100)
##grid = GridSearchCV(KernelDensity(kernel='gaussian'),
##                    {'bandwidth': bandwidths},
##                    cv=5,
##                    verbose = 1)
##grid.fit(Zarr[:, None]);
##print('Best params:',grid.best_params_)
#
##kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], 
#kde = KernelDensity(bandwidth=1, 
#                    kernel='gaussian')
#kde.fit(Zarr[:, None])
#
#
#### Plots
## Remove large values for ease of plotting
#Zarr = Zarr[(Zarr < 100) & (Zarr > -100)]
#x_d = np.linspace(-100,100,1000)
#cfit = cauchy.pdf(x_d,loc=loc,scale=sca)
#nfit = norm.pdf(x_d,loc=locnorm,scale=scanorm)
#tfit = t.pdf(x_d,df=dft,loc=loct,scale=scat)
#
#logprob_kde = kde.score_samples(x_d[:, None])
#
#pdf_cmb_array = []
#for x in x_d:
#    pdf_cmb_array.append(1/ncomb * pdf_cmb(x))
##    pdf_cmb_array.append(pdf_cmb(x))
#
#pdf_cmb_array = np.array(pdf_cmb_array)
#
#_ = plt.hist(Zarr,bins=100,normed=True,histtype='step')
#plt.plot(x_d,cfit,'k-') # Cauchy fit
#plt.plot(x_d,nfit,'k--') # Normal fit
##plt.plot(x_d,tfit,'k-.') # Student-t fit
#
#plt.plot(x_d,pdf_cmb_array,'r--') # Mixture
#plt.fill_between(x_d, np.exp(logprob_kde), alpha=0.5)




#avec = np.arange(20,200,20)
#for alpha in avec:
#    fit = alpha/2 * 1/np.cosh(alpha*x_d)**2
#    plt.plot(x_d,fit)
    

# _ = plt.hist(dToT[(dToT<0.1)&(dToT>-0.1)],bins=100,normed=True,histtype='step')

#bandwidths = 1e-2 * np.linspace(1,10,10)
#grid_dToT = GridSearchCV(KernelDensity(kernel='gaussian'),
#                    {'bandwidth': bandwidths},
#                    cv=5,
#                    verbose = 1)
#grid_dToT.fit(dToT[:, None]);
#print('dToT best params:',grid_dToT.best_params_)

#
#wl_sub_vec = wl_vec[pix_sub_vec]
##chosen_pix =po. choose_pixels(pix_sub_vec,bin_method='average')
#chosen_pix = np.arange(50,2951,10)
#cmb_pix = po.generate_combinations(chosen_pix,pix_sub_vec)
#
#bins = pix_vec[0::sc.pix_slice]
#wl_sub_vec = wl_vec[pix_sub_vec]
#
## Minimum and maximum wavelengths
#wl_min = np.min(wl_sub_vec)
#wl_max = np.max(wl_sub_vec)
#
## Which wavelengths are associated with the pixel combinations?
#wl_v0 = wl_vec[cmb_pix[:,0]]
#wl_v1 = wl_vec[cmb_pix[:,1]] 
#
## Create the [lambda_min,lambda_max] pairs that delimit a "bin"
#wl_binm = wl_vec[bins]
#wl_binM = wl_vec[bins[1::]]
#wl_binM = np.append(wl_binM,wl_vec[-1])
#
#### Calculate intensity ratio
#print("Calculate logR")
#logR = tf.calculate_logR(data_spl, wl_v0, wl_v1)
##poly_coeff = np.array([w_b,w_m])
##domain = np.array([wl_min,wl_max])
##pol =  Polynomial(poly_coeff,domain)
##
### Calculate the emissivities at the corresponding wavelengths
##eps1 = polynomial.polyval(wl_v1,pol.coef)
##eps0 = polynomial.polyval(wl_v0,pol.coef)
#
## No emissivity error, so we can calculate eps1 and eps0 directly
## from the given emissivity function
#eps0 = f_eps(wl_v0,1)
#eps1 = f_eps(wl_v1,1)
#
#print("Calculate Tout")
#invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
#Tout = 1/invT
#Tout *= sc.C2 * ( 1/wl_v1 - 1/wl_v0)
#
## Filter out some crazy values
#Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout, method = 'dispersion')
#
#### Distribution mixture
#print("Calculate distribution mixture")
## Standard deviation of each normal distribution
#sigma_I = 0.1 # Error on intensity
#std_array = np.sqrt(2) * np.abs(T / (sc.C2*(1/wl_v1-1/wl_v0)) * sigma_I)

### Form a dictionary of distributions
##distributions = []
##L = len(wl_v1)
##for k in range(L):
##    if(np.mod(k,10000)==0):
##        print(k)
##    dist_type = np.random.normal
##    dist_args = {"loc":0,"scale":std_array[k]}
##    distributions.append({"type":dist_type,"kwargs":dist_args})
##    
##coefficients = 1/L*np.ones(L)
###coefficients = 1/std_array**2
##coefficients /= coefficients.sum()      # in case these did not add up to 1
##sample_size = 10000
##
##num_distr = len(distributions)
##data = np.zeros((sample_size, num_distr))
##for idx, distr in enumerate(distributions):
##    data[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
##    
##random_idx = np.random.choice(np.arange(num_distr), 
##                              size=(sample_size,), 
##                              p=coefficients)
##
##sample = data[np.arange(sample_size), random_idx]
##sample_lo = sample[(sample>-0.5) & (sample<0.5)]
#dToT = (Tout-T)/T
#dToT_lo = dToT[(dToT > -0.5) & (dToT < 0.5)]
#dToT_ds = np.random.choice(dToT_lo,size=sample_size) # Downsample to sample_size samples
#
#
#### Kernel density
## Find the best bandwidth for a Gaussian kernel density
#print("Calculate best kernel density bandwidth")
#bandwidths = 10 ** np.linspace(-3, -2, 100)
#grid_dToT = GridSearchCV(KernelDensity(kernel='gaussian'),
#                    {'bandwidth': bandwidths},
#                    cv=5,
#                    verbose = 1)
#grid_dToT.fit(dToT_ds[:, None]);
#print('dToT best params:',grid_dToT.best_params_)
#
##bandwidths = 10 ** np.linspace(-3, -1, 100)
##grid_sample = GridSearchCV(KernelDensity(kernel='gaussian'),
##                    {'bandwidth': bandwidths},
##                    cv=5,
##                    verbose = 1)
##grid_sample.fit(sample_lo[:, None]);
##print('Sample best params:',grid_sample.best_params_)
#
#
### Instantiate and fit the KDE model
#print("Instantiate and fit the KDE model")
#kde_dToT = KernelDensity(bandwidth=grid_dToT.best_params_['bandwidth'], 
#                    kernel='gaussian')
#kde_dToT.fit(dToT_lo[:, None])
#
##kde_sample = KernelDensity(bandwidth=grid_sample.best_params_['bandwidth'], 
##                    kernel='gaussian')
##kde_sample.fit(sample_lo[:, None])
#
## Score_samples returns the log of the probability density
#x_d = np.linspace(-0.02,0.02,1000)
#logprob_dToT = kde_dToT.score_samples(x_d[:, None])
##logprob_sample = kde_sample.score_samples(x_d[:, None])
#
#### Plots
#plt.hist(dToT_lo,bins=100,normed=True,histtype='step')
#
#plt.fill_between(x_d, np.exp(logprob_dToT), alpha=0.5)
#plt.plot(dToT, np.full_like(dToT, -0.01), '|k', markeredgewidth=1)
#
##plt.fill_between(x_d, np.exp(logprob_sample), alpha=0.5)
##plt.plot(sample, np.full_like(sample, -0.02), '|b', markeredgewidth=1)
#
#plt.ylim(-0.02, 50)
#plt.xlim(-1,1)