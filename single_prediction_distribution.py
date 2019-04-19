import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,ncx2
import spectropyrometer_constants as sc
import generate_spectrum as gs
import temperature_functions as tf

import pixel_operations as po

from statistics import tukey_fence

from scipy.interpolate import splev,splrep
from scipy.special import factorial2

nthat = 3000 # 3000 samples
dlambda = 100
chosen_pix = np.arange(50,2950,dlambda)


nwl = len(chosen_pix)
nwl = (int)(nwl * (nwl-1)/2)

# Distribution of all the dT over T
dToT_dist = np.zeros((nwl,nthat))
Tden_dist = np.zeros((nwl,nthat))
Ii_dist = np.zeros((nwl,nthat))
Ij_dist = np.zeros((nwl,nthat))
Iinoisy_dist = np.zeros((nwl,nthat))
Ijnoisy_dist = np.zeros((nwl,nthat))
Tave_dist = np.zeros(nthat)

T = 3000
w_wl = np.array([300,350,400,500,600,700,800,900])
w_eps_data = np.array([0.474,0.473,0.474,0.462,0.448,0.436,0.419,0.401])

w_m,w_b = np.polyfit(w_wl,w_eps_data,deg=1)
w_eps = lambda wl,T: w_m*wl + w_b

# Black and gray body
bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
gr_eps = lambda wl,T: 0.1 * np.ones(len(wl))


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
f_eps = art_eps


for idx in range(nthat):
    if(np.mod(idx,500)==0):
        print(idx)
    
    I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
            wl_vec,T,pix_vec,f_eps)
    wl_sub_vec = wl_vec[pix_sub_vec]
    #chosen_pix =po. choose_pixels(pix_sub_vec,bin_method='average')
    chosen_pix = np.arange(50,2950,dlambda)
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
#    logR = tf.calculate_logR(data_spl, wl_v0, wl_v1)
    logR_array = []
    logIi = filtered_data[cmb_pix[:,0]-sc.window_length]
    logIj = filtered_data[cmb_pix[:,1]-sc.window_length]
#    for wl0, wl1 in zip(wl_v0, wl_v1):                
        # Corresponding data from the filtered data
#        res0 = 10**splev(wl0, data_spl)
#        res1 = 10**splev(wl1, data_spl)
      
        # Ratio of intensities
#        R = res0/res1
#        logR = np.log(R)
    logR = np.log(10)*(logIi-logIj)
#        logR_array.append(logR)
        
#    logR_array = np.array(logR_array)    
    
    
    # No emissivity error, so we can calculate eps1 and eps0 directly
    # from the given emissivity function
    eps0 = f_eps(wl_v0,1)
    eps1 = f_eps(wl_v1,1)
    
#    print("Calculate Tout")
    invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
    Tout = 1/invT
    Tout *= sc.C2 * ( 1/wl_v1 - 1/wl_v0)
    
    # Filter out some crazy values
    Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout, method = 'dispersion')
    
    ### PDistributions
    dToT_dist[:,idx] = (Tout-T)/T
#    Tden_dist[:,idx] = 1/invT
    Tden_dist[:,idx] = (Tout/T)
#    Tden_dist[:,idx] = invT
##    Ii_dist[:,idx] = 10**splev(wl_v0,data_spl)
##    Ij_dist[:,idx] = 10**splev(wl_v1,data_spl)
#    Ii_dist[:,idx] = filtered_data[cmb_pix[:,0]-sc.window_length]
#    Ij_dist[:,idx] = filtered_data[cmb_pix[:,1]-sc.window_length]
#    Iinoisy_dist[:,idx] = noisy_data[cmb_pix[:,0]]
#    Ijnoisy_dist[:,idx] = noisy_data[cmb_pix[:,1]]
#    
#    Tave_dist[idx] = Tave
    Tave_dist[idx] = (np.mean(Tout) - T)/T
    
    ### TODO
    ### CHECK THE DISTRIBUTION FOR TAVERAGE AND FOR THE T_HATs

# Form a dictionary of distributions
print("Forming mixture")
L = len(wl_v1)
sigma_I = 0.1


#  Build mud for all wavelengths
mud_all = np.zeros(L)
for idx in range(L):
    pixi = cmb_pix[idx,0]
    pixj = cmb_pix[idx,1]
    
    # window size over 2
    wo2 = (int)(sc.window_length/2)
    
    icontrib = np.sum(np.log(I_calc)[pixi-wo2:pixi+wo2+1])
    jcontrib = np.sum(np.log(I_calc)[pixj-wo2:pixj+wo2+1])
    
    mud_all[idx] = 1/(sc.window_length+1) * (icontrib - jcontrib)

mud_all += - 5*np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
sigd_all = np.ones_like(mud_all) * np.sqrt(2/sc.window_length)*sigma_I
#sigd_all *=   / sc.window_length

lam_all = (mud_all / sigd_all)**2
    
Teq = sc.C2*(1/wl_v1-1/wl_v0) / mud_all


muThat_all = np.zeros_like(mud_all)

def mu_expansion(lam,Nseries):
    approx = 0
    for k in range(Nseries):
        approx += 1/lam**k * factorial2(2*k-1)
        
    return approx

findN = lambda N,idx: np.abs((np.abs(factorial2(2*N-1) * sigd_all[idx]**(2*N)/mud_all[idx]**(2*N+1))))
findNarray = np.zeros_like(mud_all,dtype=np.int64)
Narr = np.arange(0,30,1)
Narr = np.array(Narr,dtype=np.int64)
for idx in range(L):
    argmin = np.argmin(findN(Narr,idx))
    findNarray[idx] = (int)(Narr[argmin])
#muThat_all = Teq * (1+1/lam_all+3/lam_all**2 + 15/lam_all**3 )
    muThat_all[idx] = mu_expansion(lam_all[idx],findNarray[idx])
    
muThat_all *= Teq / T
sigdThat_all = Teq / T * np.abs(sigd_all / mud_all) * np.sqrt(1+2*(sigd_all / mud_all)**2)

lamThat_all = muThat_all**2/sigdThat_all**2
#sigdThat_all = 1/lam_all * (1+1/lam_all)
#sigdThat_all *= (Teq/T)**2
#sigdThat_all *= np.sqrt(4/sc.window_length) 
#sigdThat_all *= sigma_I * T / np.abs( sc.C2*(1/wl_v1-1/wl_v0))



sigdTbar = np.sum(sigdThat_all**2)
#sigdTbar = np.sum(sigdThat_all[mud_all < -0.5]**2)
sigdTbar = np.sqrt(sigdTbar)
sigdTbar /= L

cnt,bins,_ = plt.hist( Tave_dist[(Tave_dist<0.1) & (Tave_dist>-0.1)],bins=1000,normed='True',histtype='step')
# Account for offset
mu,sig = norm.fit(Tave_dist[(Tave_dist<0.1) & (Tave_dist>-0.1)]) 
#mu = 0
_ = plt.hist( norm.rvs(loc=mu,scale=sigdTbar,size=10000),bins=bins,normed='True',histtype='step')

def doublesum(Nwl,lam_m,dlam):
    s = 0.0
    for i in np.arange(1,Nwl,1):
        lambdai = lam_m + (i-1)*dlam
        for j in np.arange(i+1,Nwl+1,1):
            lambdaj = lambdai + j * dlam
            
            s += (lambdaj * lambdai / (j*dlam))**2 
    
    return s

            

#def sigdT(lam_m,lam_M,dlam,Nwl,T0):
def sigdT(wl_v0,wl_v1,Nwl,T0):  
    fac = 2*sigma_I / np.sqrt(sc.window_length+1) * T0 / sc.C2 * 1/Nwl
    print(fac)
    
    s = np.sqrt(np.sum((1/(1/wl_v0 - 1/wl_v1))**2))
#    s = doublesum(Nwl,lam_m,dlam)
    print(s,np.sqrt(s))
    
    return fac * np.sqrt(s)
    
    
