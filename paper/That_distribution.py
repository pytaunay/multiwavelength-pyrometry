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
The graph shows the effect of the test emissivity on the distribution of the
two-color predictions of temperature. 
We assume a correct emissivity of 0.5.
'''

### Impots
# Numpy, matplotlib, scipy
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.stats import cauchy
from scipy.stats import iqr

import itertools
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

C1 = 1.191e16 # W/nm4/cm2 Sr
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


def tukey_fence(Tvec, method='cv'):
    '''
    Function: tukey_fence
    Descritpion: Removes outliers using Tukey fencing
    Inputs:
        - Tvec: some vector
        - method: a keyword for a metric to evaluate the data dispersion. It
        can either be 
            1. 'cv' (default) to calculate the coefficient of variation which
            is defined as standard_deviation / mean, or
            2. 'dispersion' to calculate the interquartile dispersion which is
            defined as (Q3-Q1)/(Q3+Q1)
    Outputs:
        - Average of vector w/o outliers
        - Standard deviation of vector w/o outliers
        - Standard error of vector w/o outliers (%)
        - Vector w/o outliers
    '''      
    ### Exclude data w/ Tukey fencing
    T_iqr = iqr(Tvec)
    T_qua = np.percentile(Tvec,[25,75])
    
    min_T = T_qua[0] - 1.5*T_iqr
    max_T = T_qua[1] + 1.5*T_iqr
    
    T_left = Tvec[(Tvec>min_T) & (Tvec<max_T)]
    
    ### Calculate standard deviation, average
    Tstd = np.std(T_left)
    Tave = np.mean(T_left)
    
    ### Calculate a metric
    if method == 'cv':
        Tcv = Tstd/Tave*100
        metric = Tcv
    elif method == 'dispersion':
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
        Tave, Tstd, Tmetric, Tleft = tukey_fence(That, method = 'dispersion')
        
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
plot_cauchy = True # Enable / disable the plot of the Cauchy distribution fit
test_emissivity_order = 1 # Control the order of the emissivity model

# Noise
sigma_I = 0.1

# Temperature
T0 = 3000 # K

# Wavelengths
wlrange = 1.46 # wavelength range
lambda1 = 300 # nm
lambdaN = (1+wlrange)*lambda1 # nm
nwl = 50 # number of wavelengths
wdw = 51 # window size
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
neps = 5 # Total number of emissivity function to try and display
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
    
### Generate the That distributions
data = generate_That_distributions(sigma_I,T0,wl_vec,nwl,wdw,wdwo2,
                                   f_eps_true,f_eps_test,neps)

fig, ax = plt.subplots()
axins = inset_axes(ax, width="30%", height="30%", loc=2, borderpad=5)

### For all emissivities...
maxpdf = 0
for idx,dist in enumerate((data['distall'].T)[:-1]):
    f_eps = lambda wl,T: f_eps_test(idx,wl,T)
    Tave = data['Tbar'][idx]
       
    dist_filt = np.copy(dist)
    dist_filt *= Tave
    dist_filt += Tave
    Tave, Tstd, Tmetric, Tleft = tukey_fence(dist_filt, method = 'dispersion') 

    dist_filt = np.copy(Tleft)
    dmin = np.min(dist_filt)
    dmax = np.max(dist_filt)
    
    if len(dist_filt) < 2 or Tave < 0:
        continue
    
    x_d = np.linspace(dmin,dmax,500)
    
    ### Find the best bandwidth for KDE
    bandwidths = 10 ** np.linspace(0,3,200)
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

    logprob_kde = kde.score_samples(x_d[:, None])
    pdfkde = np.exp(logprob_kde)
    
        
    ### Fit a Cauchy distribution 
    loc,scale = cauchy.fit(dist_filt)
    ncauchy = cauchy.pdf(x_d,loc=loc,scale=scale)
    
    ### Print info and plot
    print(idx,dmin,dmax,np.abs(np.mean(dist)),grid.best_params_['bandwidth'],data['metric'][idx])
    p = ax.plot(x_d,pdfkde)
    axins.plot(wl_vec,f_eps(wl_vec,1))
    
    if plot_cauchy:
        ax.plot(x_d,ncauchy,linestyle='dashed',color=p[-1].get_color())
    
    idxM = np.argmax(pdfkde)
    ax.text(x_d[idxM],pdfkde[idxM],data['metric'][idx])
    
    ### Maximum of all of the PDFs
    maxpdf = max(maxpdf,np.max(pdfkde))
    maxpdf = max(maxpdf,np.max(ncauchy))

### Indicate true temperature 
ax.text(1.5 * T0, 0.8 * maxpdf, "True temperature (K): " + str(T0))
ax.set_ylim([0,maxpdf])
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Probability density function')

axins.set_xlabel("Wavelength (nm)")
axins.set_ylabel("Emissivity")
