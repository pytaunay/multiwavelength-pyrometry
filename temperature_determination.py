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
# Source: https:/github.com/pytaunay/ILX526A


import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy.interpolate import splrep,splev,UnivariateSpline
from scipy.stats import iqr
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit,minimize

C1 = 1.191e16 # W/nm4/cm2 Sr
C2 = 1.4384e7 # nm K

pix_slice = 300

'''
Function: wien_approx
Calculates the Wien approximation to Planck's law for non-constant emissivity
Inputs:
    - lnm: wavelength in nm
    - T: temperature in K
    - f_eps: a lambda function representing the emissivity as function of
    temperature and wavelength
'''
def wien_approx(lnm,T,f_eps):
    eps = f_eps(lnm,T) # Emissivity
    
    return eps * C1 / lnm**5 * np.exp(-C2/(T*lnm))

'''
Function: f_target
Target function for non-linear optimization to find m and b
Inputs:
    - m,b: parameters for the emissivity (assumed linear)
    - T: the temperature
    - lnm_vec: vector of wavelengths
    - pix: pixel numbers to consider
''' 
def f_target(X,pix,lnm_vec):
    T = X[0]
    m = X[1]
    b = X[2]
    
    p0 = pix[:,0]
    p1 = pix[:,1]
    
    l0 = lnm_vec[p0]
    l1 = lnm_vec[p1]
    
    
    t = 5 * np.log(l1/l0)
    t += np.log(m*l0+b) - np.log(m*l1+b)
    t -= C2/T*(1/l0-1/l1)
    
    return t

#def std_target(X,pix,l1,l0,Rbar):
#    gam = X[0]
#    
#    p0 = pix[:,0]
#    p1 = pix[:,1]
#    
#    l0 = lnm_vec[p0]
#    l1 = lnm_vec[p1]
#    
#    # Calculate the temperature
#    T = np.log(Rbar) - 5*np.log(l1/l0) - np.log(l0+gam) + np.log(l1+gam)
#    T = 1/T
#    T *= C2 * (1/l1-1/l0)
#    
#    # Standard deviation
#    std = np.std(T)
#    
#    return std

def T_target(X,logR,l1,l0):  
    bom = X
    
    invt = logR - 5 *np.log(l1/l0) - np.log((l0+bom)/(l1+bom))
#    print(invt)
    
    Tout = 1/invt
    Tout *= C2 * ( 1/l1 - 1/l0)

    T_iqr = iqr(Tout)
    T_qua = np.percentile(Tout,[25,75])
        
    min_T = T_qua[0] - 1.25*T_iqr
    max_T = T_qua[1] + 1.25*T_iqr
        
    T_left = Tout[(Tout>min_T) & (Tout<max_T)]
    
    ret = np.average(T_left)
    std = np.std(T_left)
    
    return ret,std

def min_target(X,logR,l1,l0):
#    m = X[0]
#    b = X[1]
#    
#    invt = logR - 5 *np.log(l1/l0) - np.log(m*l0+b) + np.log(m*l1+b)
##    print(invt)
    
    bom = X
    invt = logR - 5 *np.log(l1/l0) - np.log( (l0+bom)/(l1+bom))
    
    t = 1/invt
    t *= C2 * ( 1/l1 - 1/l0)
    
    ret = np.std(t)
#    print(m,b,ret)
    
    if np.isnan(ret):
        return 1e5
    else:   
        return ret
    
def eps_piecewise(X,lnm,lnm_binm,lnm_binM):
    # X contains each pair of [m,b]
#    mvec = X[0::2]
#    bvec = X[1::2]
    bvec = np.copy(X)
    
    ret = []
    for single_lnm in lnm:
        boolcond = (single_lnm > lnm_binm) & (single_lnm <= lnm_binM)
            
        # If we have only "False" then we are at the first pixel
        if(len(np.unique(boolcond)) == 1):
#            tmp = mvec[0] * single_lnm + bvec[0]
            tmp = bvec[0]
            tmp = np.array(tmp)
        else:
#            tmp = mvec * single_lnm + bvec
            tmp = np.copy(bvec)
            tmp *= boolcond
    
        # Only one bin is active, the rest is zeros. Summing gets rid of that one
        tmp = np.sum(tmp)    
    
#        if tmp > 1:
#            tmp = 1
#        elif tmp < 0:
#            tmp = 0
        
        if tmp > 1 or tmp < 0:
            return np.array([-1])
    
        ret.append(tmp)
    

    return np.array(ret)
    
def min_multivariate(X,logR,l1,l0,lnm_binm,lnm_binM,lnm_vec):
    eps1 = eps_piecewise(X,l1,lnm_binm,lnm_binM)
    eps0 = eps_piecewise(X,l0,lnm_binm,lnm_binM)

    if eps1[0] == -1 or eps0[0] == -1:
        return 1e5

    invt = logR - 5 *np.log(l1/l0) - np.log(eps0/eps1)
    
    t = 1/invt
    t *= C2 * ( 1/l1 - 1/l0)
    
    ret = np.std(t)
    
    ### Also calculate deviation from the data curve
    bb_eps = lambda lnm,T: 1.0 # Black body
    Tave = np.average(t)
    epsvec = eps_piecewise(X,lnm_vec,lnm_binm,lnm_binM)
    test_curve = epsvec*wien_approx(lnm_vec,Tave,bb_eps)
    
    stdcurve = np.std(np.abs(filtered_data-test_curve))
    
    if np.isnan(ret):
        return 1e5
    else:   
        return max(ret,stdcurve)  
    

def T_multivariate(X,logR,l1,l0,lnm_binm,lnm_binM):  
    eps1 = eps_piecewise(X,l1,lnm_binm,lnm_binM)
    eps0 = eps_piecewise(X,l0,lnm_binm,lnm_binM)
    
    invt = logR - 5 *np.log(l1/l0) - np.log(eps0/eps1)
#    print(invt)
    
    Tout = 1/invt
    Tout *= C2 * ( 1/l1 - 1/l0)

    T_iqr = iqr(Tout)
    T_qua = np.percentile(Tout,[25,75])
        
    min_T = T_qua[0] - 1.25*T_iqr
    max_T = T_qua[1] + 1.25*T_iqr
        
    T_left = Tout[(Tout>min_T) & (Tout<max_T)]
    
    ret = np.average(T_left)
    std = np.std(T_left)
    
    return ret,std

def lsq_multivariate(X,logR,l1,l0,lnm_binm,lnm_binM):
    eps1 = eps_piecewise(X,l1,lnm_binm,lnm_binM)
    eps0 = eps_piecewise(X,l0,lnm_binm,lnm_binM)

    if eps1[0] == -1 or eps0[0] == -1:
        return 1e5

    invt = logR - 5 *np.log(l1/l0) - np.log(eps0/eps1)
    
    t = 1/invt
    t *= C2 * ( 1/l1 - 1/l0)
    
    return t

# Tungsten 2000 K emissivity
w_lnm = np.array([300,350,400,500,600,700,800,900])
w_eps_data = np.array([0.474,0.473,0.474,0.462,0.448,0.436,0.419,0.401])

w_m,w_b = np.polyfit(w_lnm,w_eps_data,deg=1)
w_eps = lambda lnm,T: w_m*lnm + w_b
#w_spline = splrep(w_lnm,w_eps_data)
#w_eps = lambda lnm,T: splev(lnm,w_spline)  

art_lnm = np.array([300,500,1100])
art_eps_data = np.array([1,0.3,1])
art_fac = np.polyfit(art_lnm,art_eps_data,deg=2)

a0,a1,a2 = art_fac
art_eps = lambda lnm,T: a0*lnm**2 + a1*lnm + a2

bb_eps = lambda lnm,T: 1.0 # Black body
gr_eps = lambda lnm,T: 0.1 # Grey body


T_vec = np.array([1000,2000,3000])
lnm_vec = np.linspace(300,1100,(int)(3000))
pix_vec = np.linspace(0,2999,3000)
pix_vec = np.array(pix_vec,dtype=np.int64)

for T in T_vec:      
    if T == 2000:
        print(T)
        
#        I_calc = wien_approx(lnm_vec,T,art_eps)
        I_calc = wien_approx(lnm_vec,T,w_eps)
        noisy_data = np.random.normal(I_calc,0.1*I_calc)
#        noise = 0.0
        
#        res = I_calc + noise
        ### Filter data
        window_length = (int)(pix_slice/5)
        if window_length % 2 == 0:
            window_length += 1
        
        filtered_data = savgol_filter(noisy_data,window_length,3)
        

        ### Fit a line through the noise with some smoothing
        spl = splrep(lnm_vec,np.log10(filtered_data))
#        spl  = UnivariateSpline(lnm_vec,np.log(res),s=1e-3)
    
        ### Bin the pixels
        bins = pix_vec[0::pix_slice]
        
        ### For each bin, find a pixel
        rand_pix = []
        for idx in bins[:-1]:
            # Find the low and high bounds of that slice
            lo = pix_vec[idx]
            hi = pix_vec[idx + pix_slice - 1]
            
            # Pick a pixel at random in it
            pix_idx = np.random.choice(pix_vec[lo:hi],size=1)[0]
#            pix_idx = (int)(np.average([lo,hi]))
            
            rand_pix.append(pix_idx)
            
        rand_pix = np.array(rand_pix)
#            print(rand_pix)
        
#        rand_pix = np.random.choice(pix_vec,size=100,replace=False)

        
#        c_pix = it.combinations(rand_pix,2)

        Tout = []
        
        c_pix_array = []
        
        logR_array = []

#        for pix in c_pix:

#        for pix in pix_vec[1::]:
        for rpix in rand_pix:
            
            p1vec_p = pix_vec[rpix::pix_slice]
            p1vec_m = pix_vec[rpix::-pix_slice]
            
            p1vec = np.concatenate((p1vec_m,p1vec_p))
            p1vec = np.unique(p1vec)
            p1vec = p1vec[p1vec != rpix]
        
            p0 = rpix
                       
            for p1 in p1vec:                
                l0 = lnm_vec[p0]
                l1 = lnm_vec[p1]
                

                res0 = 10**splev(l0,spl)
                res1 = 10**splev(l1,spl)
#                res0 = np.exp(spl(l0))
#                res1 = np.exp(spl(l1))
                
#                if res0 < 0 or res1 < 0:
#                res0 = res[p0]
#                res1 = res[p1]         
                    
                R = res0/res1
                
                try:
                    Ttarget = C2 * ( 1/l1 - 1/l0) / (np.log(R)-5*np.log(l1/l0))
                except:
                    continue
                
                if Ttarget < 0 or np.isnan(Ttarget):
                    continue
                      
                Tout.append(Ttarget)
                c_pix_array.append((p0,p1))
                logR_array.append(np.log(R))
        
        c_pix_array = np.array(c_pix_array)
        Tout = np.array(Tout)      
        logR_array = np.array(logR_array)
        
        #print(np.mean(Tout),np.std(Tout,ddof=1))  
#        plt.plot(Tout)
#        plt.ylim([0,3000])
        
        # Exclude data w/ Tukey fencing
        T_iqr = iqr(Tout)
        T_qua = np.percentile(Tout,[25,75])
        
        min_T = T_qua[0] - 1.25*T_iqr
        max_T = T_qua[1] + 1.25*T_iqr
        
        T_left = Tout[(Tout>min_T) & (Tout<max_T)]
        
        
        std = np.std(T_left)
        Tave = np.mean(T_left)
        rse = std/Tave
    
        print(Tave,std,rse*100)
        
       
        ### No consensus
        # TODO: change l to an actual wavelength
        # TODO: Make sure m,b are well defined and bounded
        # TODO: Move f to its own function
        if rse*100 > 0.75:
#            X0 = np.array([-1e-4,0.5])
#            lnm_vec0 = lnm_vec[c_pix_array[:,0]]
#            lnm_vec1 = lnm_vec[c_pix_array[:,1]]
#            
#            f = lambda X: min_target(X,logR_array,lnm_vec1,lnm_vec0)
##            minimize(f,X0)
#            sol = minimize(f,X0,method='L-BFGS-B',bounds=[(-1e-3,1e-3),(1e-5,1.0)])
            
#            X0 = np.array([-1e6]) # WORKS FOR FIRST ORDER
            lnm_vec0 = lnm_vec[c_pix_array[:,0]]
            lnm_vec1 = lnm_vec[c_pix_array[:,1]]    
#            f = lambda X: min_target(X,logR_array,lnm_vec1,lnm_vec0)
            lnm_binm = lnm_vec[bins]
            lnm_binM = lnm_vec[bins[1::]]
            lnm_binM = np.append(lnm_binM,lnm_vec[-1])
            
            f = lambda X: min_multivariate(X,logR_array,lnm_vec1,lnm_vec0,lnm_binm,lnm_binM,lnm_vec)
            
#            X0 = np.zeros(2*len(lnm_binM))
#            X0[1::2] = 0.5
            X0 = 0.5 * np.ones(len(lnm_binm))
            min_options = {'xatol':1e-8,'fatol':1e-8,'maxfev':300}
            sol = minimize(f,X0,method='Nelder-Mead',options = min_options)
            
            #####
            ##### TRYING WITH BOUNDED BFGS AND TNC WITH PIECEWISE CONSTANT
#            bounds_list = []
#            all_bounds = (1e-4,1)
#            for elem in X0:
#                bounds_list.append(all_bounds)
            
#            min_options = {'ftol' : 1e2 * np.finfo(float).eps,'eps':1e1*np.finfo(float).eps}
#            sol = minimize(f,X0,method='L-BFGS-B',bounds=bounds_list,options=min_options)

#            min_options = {'ftol' : 1e1 * np.finfo(float).eps,'xtol':1e1*np.finfo(float).eps,
#                           'maxiter':1000}
#            sol = minimize(f,X0,method='TNC',bounds=bounds_list,options=min_options)

            
#            m = sol.x[0]
#            b = sol.x[1]
#            unk_eps = lambda lnm,T: m*lnm+b
#            
#            Tave,std = T_target(sol.x,logR_array,lnm_vec1,lnm_vec0)
            Tave,std = T_multivariate(sol.x,logR_array,lnm_vec1,lnm_vec0,lnm_binm,lnm_binM)
            print(Tave,std,std/Tave*100,sol.x)
#        
            # Then get epsilon such that we get the filtered data
            
            
#        
#            plt.semilogy(lnm_vec,noisy_data)
#            plt.semilogy(lnm_vec,filtered_data)
#            plt.semilogy(lnm_vec,reconstructed_data)
#        
#            plt.plot(w_lnm,w_eps_data,'o')
#            plt.plot(lnm_vec,unk_eps(lnm_vec,Tave))
#            plt.plot(lnm_vec,w_eps(lnm_vec,T))
                    
        bb_reconstructed = wien_approx(lnm_vec,Tave,bb_eps)
        eps_vec = filtered_data/bb_reconstructed
        
#        reconstructed_data = wien_approx(lnm_vec,Tave,unk_eps)
        reconstructed_data = bb_reconstructed * eps_vec # exactly filtered
            
#        plt.semilogy(lnm_vec,noisy_data)
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.semilogy(lnm_vec,filtered_data)
        ax1.semilogy(lnm_vec,reconstructed_data)
        
        epsret = eps_piecewise(sol.x,lnm_vec,lnm_binm,lnm_binM)
        ax1.semilogy(lnm_vec,epsret*bb_reconstructed)
#            
#        ax2.plot(w_lnm,w_eps_data,'o')
        ax2.plot(lnm_vec,eps_vec)
        ax2.plot(lnm_vec,w_eps(lnm_vec,Tave),'--')
        
        epsret = eps_piecewise(sol.x,lnm_vec,lnm_binm,lnm_binM)
        ax2.plot(lnm_vec,epsret,'-.')
#        ax2.plot(lnm_vec,art_eps(lnm_vec,Tave),'--')
        
        
#        plt.show()
#            print("No consensus")
        
        
        ### Get the new curve based on the data
#        I_new = wien_approx(lnm_vec,ave,bb_eps)
        
#        plt.plot(lnm_vec,res,'.')
#        plt.plot(lnm_vec,I_new)
        
        ### Plot the residuals
#        plt.figure()
#        residuals = np.abs(I_new - res)/res               
        
#        plt.plot(lnm_vec,residuals)
#        plt.plot(residuals)
#        std_res = np.std(residuals)
#        ave_res = np.mean(residuals)
#        rse_res = std_res/ave_res
#        print(ave_res,std_res,rse_res)
        
        
#        if np.mean(residuals) > 0.1:           
#            if np.std(residuals) / np.mean(residuals) < 0.05:
#                print("Grey body")
#            else:
#                print("Non-standard")
        
#        ssres = np.square(I_new-res).sum()
#        sstot = np.square(res-np.mean(res)).sum()
#        print(1-ssres/sstot)
#        print(mse)

#        plt.semilogy(lnm_vec,res)
#        plt.semilogy(lnm_vec,10**splev(lnm_vec,spl))
#    plt.semilogy(np.exp(spl(lnm_vec)))


