import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import spectropyrometer_constants as sc
import generate_spectrum as gs
import temperature_functions as tf

import pixel_operations as po

from statistics import tukey_fence

from scipy.interpolate import splev,splrep
from scipy.special import factorial2

from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV


def generate_Taverage_distribution(T0,wl_vec,pix_vec,dlambda,nthat):
    '''
    This function generates a distribution of errors. The errors are between
    the true temperature and the one calculated with the variance method.
    Inputs:
        - T0: true temperature
        - wl_vec: the vector of wavelengths
        - pix_vec: the vector of pixels
        - dlambda: the number of wavelengths to skip
        - nthat: number of Monte-Carlo sample to generate
    
    '''
    ### Grey body
    gr_eps = lambda wl,T: 0.5 * np.ones(len(wl))
    
    # Distribution of Taverage over T
    Tave_dist = np.zeros(nthat)
    
    ### Sample data
    for idx in range(nthat):
        if(np.mod(idx,2000)==0):
            print(idx)
        
        I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
                wl_vec,T0,pix_vec,gr_eps)
        chosen_pix = np.arange(50,2950,dlambda)
        cmb_pix = po.generate_combinations(chosen_pix,pix_sub_vec)
        
        bins = pix_vec[0::sc.pix_slice]
                
        # Which wavelengths are associated with the pixel combinations?
        wl_v0 = wl_vec[cmb_pix[:,0]]
        wl_v1 = wl_vec[cmb_pix[:,1]] 
        
        # Create the [lambda_min,lambda_max] pairs that delimit a "bin"
        wl_binM = wl_vec[bins[1::]]
        wl_binM = np.append(wl_binM,wl_vec[-1])
        
        ### Calculate intensity ratio
        logIi = filtered_data[cmb_pix[:,0]-sc.window_length]
        logIj = filtered_data[cmb_pix[:,1]-sc.window_length]
    
        logR = np.log(10)*(logIi-logIj)
        
        # No emissivity error, so we can calculate eps1 and eps0 directly
        # from the given emissivity function
        eps0 = gr_eps(wl_v0,1)
        eps1 = gr_eps(wl_v1,1)
        
        invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
        Tout = 1/invT
        Tout *= sc.C2 * ( 1/wl_v1 - 1/wl_v0)
        
        # Filter out some crazy values
        Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout, method = 'dispersion')
        
        ### Distributions
        Tave_dist[idx] = (np.mean(Tout) - T0)/T0
        
    
    ### Write data to disk
    root = 'variance_calculations/'
    np.save(root+'cmb_pix.npy',cmb_pix)
    np.save(root+'I_calc.npy',I_calc)
    np.save(root+'wl_v1.npy',wl_v1)
    np.save(root+'wl_v0.npy',wl_v0)
    np.save(root+'eps0.npy',eps0)
    np.save(root+'eps1.npy',eps1)
        
    return Tave_dist

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


def compute_high_order_variance(T0):    
    root = 'variance_calculations/'
    cmb_pix = np.read(root+'cmb_pix.npy')
    I_calc = np.read(root+'I_calc.npy')
    wl_v1 = np.read(root+'wl_v1.npy')
    wl_v0 = np.read(root+'wl_v0.npy')
    eps0 = np.read(root+'eps0.npy')
    eps1 = np.read(root+'eps1.npy')
    
    # Number of combinations
    Ncomb = len(wl_v1)
    
    # True window size
    w = sc.window_length + 1

    ### Noise on measurements    
    sigma_I = 0.1
    
    ### Denominator average and variance for all wavelengths
    # mud, sigd
    mud = np.zeros(Ncomb)
    for idx in range(Ncomb):
        pixi = cmb_pix[idx,0]
        pixj = cmb_pix[idx,1]
        
        # window size over 2
        wo2 = (int)(sc.window_length/2)
        
        icontrib = np.sum(np.log(I_calc)[pixi-wo2:pixi+wo2+1])
        jcontrib = np.sum(np.log(I_calc)[pixj-wo2:pixj+wo2+1])
        
        mud[idx] = 1/w * (icontrib - jcontrib)
    
    mud += - 5*np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
    sigd = np.ones_like(mud) * np.sqrt(2/w)*sigma_I
       
    ### muThat, sigThat
    muThat = np.zeros_like(mud)
    sigThat = np.zeros_like(mud)
    
    # Teq
    Teq = sc.C2*(1/wl_v1-1/wl_v0) / mud
    
    # Taylor expansion for muThat: we want to find the "best" order in the 
    # expansion of the mean of 1/X
    sigomud = sigd/mud
    sigomud2 = sigomud**2
    
    findN = lambda N,idx: np.abs(1/mud[idx]) * factorial2(2*N-1) * sigomud2**N
    Narr = np.arange(0,30,1)
    Narr = np.array(Narr,dtype=np.int64)
    
    for idx in range(Ncomb):
        argmin = np.argmin(findN(Narr,idx))
        order = (int)(Narr[argmin])
        
        mu = mud[idx]
        sig = sigd[idx]
        muThat[idx] = muthat_expansion(mu,sig,order)
    
    muThat *= Teq / T0
    
    # Taylor expansion for sigThat: we keep only the first two orders
    sigThat = (Teq/T0)**2 * sigomud2 * (1 + 2*sigomud2)
    sigThat = np.sqrt(sigThat)    
    
    
    ### muTbar, sigTbar
    # muTbar: subtract 1 bc. we want (Tbar-T0)/T0 = Tbar/T0 - 1
    muTbar = 1/Ncomb * np.sum(muThat) - 1
    
    # sigTbar
    sigTbar = 1/Ncomb**2 * np.sum(sigThat)
    
    return muTbar,sigTbar


#### Input parameters
#nthat = 2000 # Number of samples for Monte-Carlo
#T0 = 3000
#
## Wavelengths
#dlambda = 50 # Skip that many wavelengths
#chosen_pix = np.arange(50,2950,dlambda)
#lambda_0 = 300
#lambda_N = 1100
#
#wl_vec = np.linspace(lambda_0,lambda_N,(int)(3000))
#pix_vec = np.linspace(0,2999,3000)
#pix_vec = np.array(pix_vec,dtype=np.int64)
#
#nwl = len(chosen_pix)
#nwl = (int)(nwl * (nwl-1)/2)
#
#### Create some data
#Tbar_ds = generate_Taverage_distribution(T0,wl_vec,pix_vec,dlambda,nthat)
#
#### Calculate the variance based on the second-order accurate expansions
#muTbar, sigTbar = compute_high_order_variance()

### Calculate the variance based on the successive approximations to get an 
### analytical expression


#
#def doublesum(Nwl,lam_m,dlam):
#    s = 0.0
#    for i in np.arange(1,Nwl,1):
#        lambdai = lam_m + (i-1)*dlam
#        for j in np.arange(i+1,Nwl+1,1):
#            lambdaj = lambdai + j * dlam
#            
#            s += (lambdaj * lambdai / (j*dlam))**2 
#    
#    return s
#
#def sigdT(wl_v0,wl_v1,Nwl,T0):  
#    fac = 2*sigma_I / np.sqrt(sc.window_length+1) * T0 / sc.C2 * 1/Nwl
#     
#    s = np.sqrt(np.sum((1/(1/wl_v0 - 1/wl_v1))**2))
#    
#    return fac * s
#
#### USE KERNEL DENSITY ESTIMATE TO GENERATE NICE PLOTS
#### Kernel density
## Find the best bandwidth for a Gaussian kernel density
#print("Calculate best kernel density bandwidth")
#bandwidths = 10 ** np.linspace(-3, -2, 100)
#grid_dToT = GridSearchCV(KernelDensity(kernel='gaussian'),
#                    {'bandwidth': bandwidths},
#                    cv=5,
#                    verbose = 1)
#grid_dToT.fit(Tave_dist[:, None]);
#print('dToT best params:',grid_dToT.best_params_)
#
#bandwidths = 10 ** np.linspace(-3, -1, 100)
#grid_sample = GridSearchCV(KernelDensity(kernel='gaussian'),
#                    {'bandwidth': bandwidths},
#                    cv=5,
#                    verbose = 1)
#grid_sample.fit(sample_lo[:, None]);
#print('Sample best params:',grid_sample.best_params_)
#
#
### Instantiate and fit the KDE model
#print("Instantiate and fit the KDE model")
#kde_dToT = KernelDensity(bandwidth=grid_dToT.best_params_['bandwidth'], 
#                    kernel='gaussian')
#kde_dToT.fit(dToT_lo[:, None])
#
#kde_sample = KernelDensity(bandwidth=grid_sample.best_params_['bandwidth'], 
#                    kernel='gaussian')
#kde_sample.fit(sample_lo[:, None])
#
## Score_samples returns the log of the probability density
#x_d = np.linspace(-0.5,0.5,1000)
#logprob_dToT = kde_dToT.score_samples(x_d[:, None])
#logprob_sample = kde_sample.score_samples(x_d[:, None])
#
#### Plots
#plt.hist(dToT_lo,bins=100,normed=True,histtype='step')
#
#plt.fill_between(x_d, np.exp(logprob_dToT), alpha=0.5)
#plt.plot(dToT, np.full_like(dToT, -0.01), '|k', markeredgewidth=1)
#
#plt.fill_between(x_d, np.exp(logprob_sample), alpha=0.5)
#plt.plot(sample, np.full_like(sample, -0.02), '|b', markeredgewidth=1)
#
#plt.ylim(-0.02, 50)
#plt.xlim(-1,1)
#
#### PLOTS
##cnt,bins,_ = plt.hist( Tave_dist[(Tave_dist<0.1) & (Tave_dist>-0.1)],bins=100,normed='True',histtype='step')
#### Account for offset
##mu,sig = norm.fit(Tave_dist[(Tave_dist<0.1) & (Tave_dist>-0.1)]) 
####mu = 0
##_ = plt.hist( norm.rvs(loc=mu,scale=sigdTbar,size=10000),bins=bins,normed='True',histtype='step')
##siganalytic = sigdT(wl_v0,wl_v1,len(wl_v0),T)
##_ = plt.hist( norm.rvs(loc=mu,scale=siganalytic,size=10000),bins=bins,normed=True,histtype='step')
##
###print(len(np.arange(50,2950,dlambda)),sig)
##plt.xlim([-0.1,0.1])
##    
#
#### 
#
#
#
#    
