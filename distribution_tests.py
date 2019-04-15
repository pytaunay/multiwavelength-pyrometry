import numpy as np
from numpy.polynomial import Polynomial, polynomial
import matplotlib.pyplot as plt
from scipy.stats import norm
import spectropyrometer_constants as sc
import generate_spectrum as gs
import temperature_functions as tf

import pixel_operations as po

from statistics import tukey_fence

from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut

T = 3000
w_wl = np.array([300,350,400,500,600,700,800,900])
w_eps_data = np.array([0.474,0.473,0.474,0.462,0.448,0.436,0.419,0.401])

w_m,w_b = np.polyfit(w_wl,w_eps_data,deg=1)
w_eps = lambda wl,T: w_m*wl + w_b


### Vectors of pixels and wavelengths
wl_vec = np.linspace(300,1100,(int)(3000))
pix_vec = np.linspace(0,2999,3000)
pix_vec = np.array(pix_vec,dtype=np.int64)
f_eps = w_eps


I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
        wl_vec,T,pix_vec,f_eps)
wl_sub_vec = wl_vec[pix_sub_vec]
#chosen_pix =po. choose_pixels(pix_sub_vec,bin_method='average')
chosen_pix = np.arange(50,2951,10)
cmb_pix = po.generate_combinations(chosen_pix,pix_sub_vec)

bins = pix_vec[0::sc.pix_slice]
wl_sub_vec = wl_vec[pix_sub_vec]

# Minimum and maximum wavelengths
wl_min = np.min(wl_sub_vec)
wl_max = np.max(wl_sub_vec)

# Which wavelengths are associated with the pixel combinations?
wl_v0 = wl_vec[cmb_pix[:,0]]
wl_v1 = wl_vec[cmb_pix[:,1]] 

# Create the [lambda_min,lambda_max] pairs that delimit a "bin"
wl_binm = wl_vec[bins]
wl_binM = wl_vec[bins[1::]]
wl_binM = np.append(wl_binM,wl_vec[-1])

### Calculate intensity ratio
print("Calculate logR")
logR = tf.calculate_logR(data_spl, wl_v0, wl_v1)
poly_coeff = np.array([w_b,w_m])
domain = np.array([wl_min,wl_max])
pol =  Polynomial(poly_coeff,domain)

# Calculate the emissivities at the corresponding wavelengths
eps1 = polynomial.polyval(wl_v1,pol.coef)
eps0 = polynomial.polyval(wl_v0,pol.coef)

print("Calculate Tout")
invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
Tout = 1/invT
Tout *= sc.C2 * ( 1/wl_v1 - 1/wl_v0)

# Filter out some crazy values
Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout, method = 'dispersion')

### Standard deviation
std_array = np.sqrt(2) * np.abs(T / (sc.C2*(1/wl_v1-1/wl_v0)) * 0.1)

distributions = []
L = len(wl_v1)
for k in range(L):
    if(np.mod(k,10000)==0):
        print(k)
    dist_type = np.random.normal
    dist_args = {"loc":0,"scale":std_array[k]}
    distributions.append({"type":dist_type,"kwargs":dist_args})
    
coefficients = 1/L*np.ones(L)
coefficients /= coefficients.sum()      # in case these did not add up to 1
sample_size = 2000

num_distr = len(distributions)
data = np.zeros((sample_size, num_distr))
for idx, distr in enumerate(distributions):
    data[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)
sample = data[np.arange(sample_size), random_idx]
#plt.hist(sample, bins=1000,normed=True)
sample_lo = sample[(sample>-0.1) & (sample<0.1)]
#plt.hist(sample_lo, bins=100,normed=True)
plt.hist( (Tleft-T)/T,bins=100,normed=True,histtype='step')

dToT = (Tout-T)/T


### Kernel density
# Find the best bandwidth for a Gaussian kernel density
bandwidths = 10 ** np.linspace(-1, 0, 50)
grid_dToT = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=LeaveOneOut(len(dToT)))
grid_dToT.fit(dToT[:, None]);
print('dToT best params:',grid_dToT.best_params_)

grid_sample = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=LeaveOneOut(len(sample)))
grid_sample.fit(sample[:, None]);
print('Sample best params:',grid_sample.best_params_)


## Instantiate and fit the KDE model
kde_dToT = KernelDensity(bandwidth=grid_dToT.best_params_['bandwidth'], 
                    kernel='gaussian')
kde_dToT.fit(dToT[:, None])

kde_sample = KernelDensity(bandwidth=grid_sample.best_params_['bandwidth'], 
                    kernel='gaussian')
kde_sample.fit(sample[:, None])

# Score_samples returns the log of the probability density
x_d = np.linspace(-10,10,1000)
logprob_dToT = kde_dToT.score_samples(x_d[:, None])
logprob_sample = kde_sample.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob_dToT), alpha=0.5)
plt.plot(dToT, np.full_like(dToT, -0.01), '|k', markeredgewidth=1)

plt.fill_between(x_d, np.exp(logprob_sample), alpha=0.5)
plt.plot(sample, np.full_like(sample, -0.01), '|k', markeredgewidth=1)

plt.ylim(-0.02, 5)
plt.xlim(-5,5)



#plt.show()

#
#tout
#Tout
#plt.hist((Tout-T)/Tout,100)
#plt.hist((Tout-T)/Tout,100,normed=True)
#s2_array = np.sqrt(2)*(T/(sc.C2*(1/wl_v1-1/wl_v0)))*0.1
#s2_array
#scipy.stats.norm.rvs(loc=0,scale=np.abs(s2_array))
#len(wl_v1)
#vechist = []
#L = len(wl_v1)
#for k in range(2000):
#    tmp = scipy.stats.norm.rvs(loc=0,scale=np.abs(s2_array))
#    stmp = np.sum(tmp)
#    stmp *= 2/(L*(L-1))
#    vechist.append(stmp)
#    
#    
#vechist
#len(vechist)
#plt.hist(vechist,100,normed=True)
#plt.hist((Tout-T)/Tout,100,normed=True)
#plt.hist(vechist,100,normed=True)
#vechist = []
#L = len(wl_v1)
#for k in range(2000):
#    tmp = scipy.stats.norm.rvs(loc=0,scale=np.abs(s2_array))
#    stmp = np.sum(tmp)
#    stmp *= 1/L
#    vechist.append(stmp)
#plt.hist(vechist,100,normed=True)
#plt.hist((Tout-T)/Tout,100,normed=True)
#plt.hist(vechist,100,normed=True)
#plt.hist(vechist,100)
#plt.hist((Tout-T)/Tout,100)
#plt.hist((Tout-T)/Tout,100,normed=True)
#plt.hist(vechist,100,normed=True)
#vechist = []
#L = len(wl_v1)
#for k in range(10000):
#    tmp = scipy.stats.norm.rvs(loc=0,scale=np.abs(s2_array))
#    stmp = np.sum(tmp)
#    stmp *= 1/L
#    vechist.append(stmp)
#plt.hist(vechist,100,normed=True)
#plt.hist((Tout-T)/Tout,100,normed=True)
#plt.hist(vechist,100,normed=True)
#plt.hist((Tout-T)/Tout,100,normed=True)
#plt.hist(vechist,50,normed=True)
#plt.hist(vechist,200,normed=True)
#np.var( (Tout-3000)/3000)
#5.62e-7*3000**2
#1/L
#1/L * 2 *np.sum( 3000**2 / (sc.C2**2 * (1/wl_v1-1/wl_v0))**2)
#1/L * 2 *np.sum( 3000**2 / (sc.C2**2 * (1/wl_v1-1/wl_v0))**2) * 0.1**2
#1/L * 2 *np.sum( 1 / (sc.C2**2 * (1/wl_v1-1/wl_v0))**2) * 0.1**2
#2*1/L * np.sum( (3000*0.1/(sc.C2*(1/wl_v1-1/wl_v0)))**2)
#np.sqrt(np.var( (Tout-3000)/3000))
#np.mean( (Tout-3000)/3000)
#np.std( (Tout-3000)/3000)
#L
#len(wl_v1)
#2*1/L * np.sum( (0.1/(sc.C2*(1/wl_v1-1/wl_v0)))**2)
#plt.plot(vechist)
#plt.hist(vechist,200,normed=True)
#scipy.stats.norm.fit(vechist)
#scipy.stats.norm.fit((Tout-T)/T)
#np.var(vechist)
#np.mean(vechist)
#np.var((Tout-T)/T)
#np.mean((Tout-T)/T)
#len(Tout)
#len(wl_v1)
#Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout, method = 'dispersion')
#from statistics
#from statistics import tukey_fence
#Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout, method = 'dispersion')
#np.mean((Tleft-T)/T)
#np.var((Tleft-T)/T)
#plt.hist(Tleft,normed=True)
#plt.hist(Tleft,100,normed=True)
#plt.hist( (Tleft-T)/T,100,normed=True)
#plt.hist(vechist,100,normed=True)