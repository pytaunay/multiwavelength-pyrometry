import numpy as np
import matplotlib.pyplot as plt
import spectropyrometer_constants as sc
import generate_spectrum as gs

import pixel_operations as po

from statistics import tukey_fence

from scipy.special import factorial2
from scipy.stats import skew,kurtosis
from scipy.stats import cauchy

from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

'''
Investigate the effect of a bunch of variables on the distribution of T_hats
for a SINGLE measurement
Idea: we may be able to get a metric that gives us information about 
correctness of chosen emissivity model
'''

def generate_Taverage_distribution(T0,wl_vec,pix_vec,nwl):
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
    ### Grey body
    gr_eps = lambda wl,T: 0.5 * np.ones(len(wl))
    
    # Distribution of Taverage over T
    Tave_dist = np.zeros(nthat)
    
    ### Sample data       
    I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
            wl_vec,T0,pix_vec,gr_eps)
#        chosen_pix = np.arange(50,2950,dlambda)
    chosen_pix = np.linspace(50,2949,nwl)
    chosen_pix = np.array(chosen_pix,dtype=np.int64)
    
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
#        logIi = np.log10(noisy_data)[cmb_pix[:,0]-sc.window_length]
#        logIj = np.log10(noisy_data)[cmb_pix[:,1]-sc.window_length]

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
    Tave_dist = (np.mean(Tout) - T0)/T0
    dToT = (Tout-T0)/T0
    
    ### Write data to disk
    root = 'variance_calculations/'
    np.save(root+'cmb_pix.npy',cmb_pix)
    np.save(root+'I_calc.npy',I_calc)
    np.save(root+'wl_v1.npy',wl_v1)
    np.save(root+'wl_v0.npy',wl_v0)
    np.save(root+'eps0.npy',eps0)
    np.save(root+'eps1.npy',eps1)
    np.save(root+'filtered_data.npy',filtered_data)
        
    return Tave_dist, dToT

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


w_wl = np.array([300,350,400,500,600,700,800,900])
w_eps_data = np.array([0.474,0.473,0.474,0.462,0.448,0.436,0.419,0.401])

w_m,w_b = np.polyfit(w_wl,w_eps_data,deg=1)
w_eps = lambda wl,T: w_m*wl + w_b

# Black and gray body
bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
gr_eps = lambda wl,T: 0.5 * np.ones(len(wl))

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
#f_eps = art_eps

### Input parameters
nthat = 1000 # Number of samples for Monte-Carlo
#T0vec = np.arange(1000,3000,100)
T0vec = np.array([3000])
sigma_I = 0.1
w = sc.window_length + 1

# Wavelengths
#Rarray = np.logspace(-1,1,10)
#lambdavec = np.arange(200,1200,200)
lambdavec = np.array([300])
Rarray = np.array([2])
Narray = np.array([200])
#Narray = np.arange(10,500,100)
Narray = np.array(Narray,dtype=np.int64)
NvsR = []

res = []
for lambda_0 in lambdavec:
    for T0 in T0vec:
        for Rapprox in  Rarray:   
    #        lambda_0 = 300
            lambda_N = (1+Rapprox) * lambda_0
            
            for nwl in Narray:
                chosen_pix = np.linspace(50,2949,nwl)
                chosen_pix = np.array(chosen_pix,dtype=np.int64)
                       
                #lambda_0 = 300
                #lambda_N = 1100
        
                wl_vec = np.linspace(lambda_0,lambda_N,(int)(3000))
                
                pix_vec = np.linspace(0,2999,3000)
                pix_vec = np.array(pix_vec,dtype=np.int64)
                
                ncomb = len(chosen_pix)
                ncomb = (int)(nwl * (nwl-1)/2)
                
                _, dToT = generate_Taverage_distribution(T0,wl_vec,pix_vec,nwl)
                muTbar,sigTbar,ratio,muThat,sigThat = compute_high_order_variance(T0,sigma_I,w)
#                _ = plt.hist(dToT[(dToT<0.1)&(dToT>-0.1)],bins=100,normed=True,histtype='step')
#                dToT = dToT[(dToT<0.1)&(dToT>-0.1)]
    
                loc,sca = cauchy.fit(dToT)
                x_d = np.linspace(-0.1,0.1,1000)
                plt.plot(x_d,cauchy.pdf(x_d,loc=loc,scale=sca),'k-')
    
                mu = np.average(dToT)
                sig = np.std(dToT)
                skw = skew(dToT)
                krt = kurtosis(dToT,fisher=False)
                
                res.append([mu,sig,skw,krt])

res = np.array(res)
#plt.xlim([-0.1,0.1])





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