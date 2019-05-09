import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import cauchy, norm, t

from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

nwl = 200
ncomb = (int)(nwl * (nwl-1)/2)

alpha = 0.01
Rp1 = 1 + alpha * (nwl - 1)

sigarr = []
sigarr_expon = []
sigarr_div = []
ijarr = []

### Create a bunch of standard deviations
for ii in range(nwl-1):
    for jj in range(ii+1,nwl):
        lsig = 1/alpha * 1/(ii-jj)
        lsig *= (1+alpha*ii)
        lsig *= (1+alpha*jj)
        sigarr.append(lsig)
        ij = jj - 1 -ii + 1/2 * (2*nwl - (ii + 1)) * ii
        ijarr.append(ij)
        
        lsig = Rp1**(ii/(nwl-1)) * Rp1**(jj/(nwl-1))
        lsig /= (Rp1**(ii/(nwl-1)) - Rp1**(jj/(nwl-1)))
        sigarr_expon.append(lsig)
        
        lsig = (1-ii / (ii-(nwl+1))) * (1-jj / (jj-(nwl+1))) 
        lsig /= (1-ii / (ii-(nwl+1))) - (1-jj / (jj-(nwl+1))) 
        sigarr_div.append(lsig)
        
sigarr = np.array(sigarr)
sigarr = np.abs(sigarr)
sigarr_expon = np.array(sigarr_expon)
sigarr_expon = np.abs(sigarr_expon)
sigarr_div = np.array(sigarr_div)
sigarr_div = np.abs(sigarr_div)
ijarr = np.array(ijarr)

### Sample the nwl * (nwl-1)/2 normal distributions
Zarr = np.zeros(ncomb)
for m in range(ncomb):
    sigrand = sigarr[m]
#    sigrand = 10
#    sigrand = sigarr_expon[m]
#    sigrand = sigarr_div[m]
#    sigrand = np.random.choice(sigarr_expon)
    Xrand = norm.rvs(loc=0,scale=sigrand,size=1)
    Zarr[m] = Xrand

### Fit a Cauchy distribution
loc,sca = cauchy.fit(Zarr)
locnorm, scanorm = norm.fit(Zarr)
dft, loct, scat = t.fit(Zarr)

### Compound distribution
#sigarr[:] = sigrand
#weights = 1/sigarr_expon
#weights = weights / np.sum(weights)
weights = np.ones_like(sigarr)
pdf_cmb = lambda x: np.sum(weights * 1/sigarr * 1/np.sqrt(2*np.pi) * np.exp(-1/2*x**2/sigarr**2))
#pdf_cmb  = lambda x: np.sum(weights * 1/sigarr_expon * 1/np.sqrt(2*np.pi) * np.exp(-1/2*x**2/sigarr_expon**2))
#pdf_cmb  = lambda x: np.sum(weights * 1/sigarr_div * 1/np.sqrt(2*np.pi) * np.exp(-1/2*x**2/sigarr_div**2))

### Buhlmann
#v2 = np.var(sigarr)



### KDE
print("KDE")
#bandwidths = 10 ** np.linspace(-3, -2, 100)
#grid = GridSearchCV(KernelDensity(kernel='gaussian'),
#                    {'bandwidth': bandwidths},
#                    cv=5,
#                    verbose = 1)
#grid.fit(Zarr[:, None]);
#print('Best params:',grid.best_params_)

#kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], 
kde = KernelDensity(bandwidth=1, 
                    kernel='gaussian')
kde.fit(Zarr[:, None])


### Plots
# Remove large values for ease of plotting
Zarr = Zarr[(Zarr < 100) & (Zarr > -100)]
x_d = np.linspace(-100,100,1000)
cfit = cauchy.pdf(x_d,loc=loc,scale=sca)
nfit = norm.pdf(x_d,loc=locnorm,scale=scanorm)
tfit = t.pdf(x_d,df=dft,loc=loct,scale=scat)

logprob_kde = kde.score_samples(x_d[:, None])

pdf_cmb_array = []
for x in x_d:
    pdf_cmb_array.append(1/ncomb * pdf_cmb(x))
#    pdf_cmb_array.append(pdf_cmb(x))

pdf_cmb_array = np.array(pdf_cmb_array)

_ = plt.hist(Zarr,bins=100,normed=True,histtype='step')
plt.plot(x_d,cfit,'k-') # Cauchy fit
plt.plot(x_d,nfit,'k--') # Normal fit
#plt.plot(x_d,tfit,'k-.') # Student-t fit

plt.plot(x_d,pdf_cmb_array,'r--') # Mixture
plt.fill_between(x_d, np.exp(logprob_kde), alpha=0.5)
