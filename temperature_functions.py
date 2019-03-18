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
from numpy.polynomial import Polynomial,polynomial

from scipy.interpolate import splev
from scipy.stats import iqr
from scipy.optimize import minimize, lsq_linear, basinhopping

from spectropyrometer_constants import C2
from spectropyrometer_constants import pix_slice,max_poly_order,rse_threshold

from goal_function import goal_function, mixed_goal_function

'''
Function: tukey_fence
Descritpion: Removes outliers using Tukey fencing
Inputs:
    - Tvec: some vector
Outputs:
    - Average of vector w/o outliers
    - Standard deviation of vector w/o outliers
    - Standard error of vector w/o outliers (%)
    - Vector w/o outliers
'''      
def tukey_fence(Tvec):
    ### Exclude data w/ Tukey fencing
    T_iqr = iqr(Tvec)
    T_qua = np.percentile(Tvec,[25,75])
    
    min_T = T_qua[0] - 1.25*T_iqr
    max_T = T_qua[1] + 1.25*T_iqr
    
    T_left = Tvec[(Tvec>min_T) & (Tvec<max_T)]
    
    ### Calculate standard deviation, average, standard error
    std = np.std(T_left)
    Tave = np.mean(T_left)
#    rse = std/Tave*100
    
    rse = (T_qua[1] - T_qua[0]) / (T_qua[1] + T_qua[0])
    

    return Tave,std,rse,T_left

'''
Function: ce_temperature
Calculates the temperature based on the averaging of multiple two-temperature
predictions and Constant Emissivity (CE)
Inputs:
    - data_spl Spline representation of the filtered intensity data
    - wl_v0, wl_v1 Vector of wavelengths chosen
Ouputs:
    - Predicted temperature from averaging (K)
    - Standard deviation (K)
    - Standard deviation (%)
    - Natural logarithm of ratio of intensities of two wavelengths. Useful for
    the non-constant emissivity case as well and avoids having to recalculate
    it.
'''
def ce_temperature(data_spl,wl_v0,wl_v1):
    Tout = []
    logR_array = []
    
    ### Standard computation
    # For each pair of wavelengths (wl0,wl1), calculate constant emissivity
    # temperature 
    for wl0,wl1 in zip(wl_v0,wl_v1):                
        # Corresponding data from the filtered data
        res0 = 10**splev(wl0,data_spl)
        res1 = 10**splev(wl1,data_spl)
      
        # Ratio of intensities
        R = res0/res1
        
        # Handle edge cases
        # Try/catch to make sure the log spits out correct values
        try:
            Ttarget = C2 * ( 1/wl1 - 1/wl0) / (np.log(R)-5*np.log(wl1/wl0))
        except:
            continue
        
        # Skip if negative or NaN
        if Ttarget < 0 or np.isnan(Ttarget):
            continue
        
        # Build vector
        Tout.append(Ttarget)
        logR_array.append(np.log(R))
    
    ### Convert to numpy arrays
    Tout = np.array(Tout)      
    logR_array = np.array(logR_array)    

    ### Returns
    Tave,std,rse,_ = tukey_fence(Tout) 
    
    return Tave,std,rse,logR_array



'''
Function: compute_temperature
Calculates the temperature
Inputs:
    - data_spl Spline representation of the filtered intensity data
    - cmb_pix Pixels chosen for each pixel bin
    - pix_vec Overall pixel vector
    - wl_vec Vector of wavelengths (nm)
Ouputs:
    - Predicted temperature from averaging (K)
    - Standard deviation (K)
    - Standard deviation (%)
    - Flag indicating if advanced method was used
'''
def compute_temperature(data_spl,cmb_pix,pix_vec,wl_vec):
    refined_fit = False
    
    bins = pix_vec[0::pix_slice]
    wl_sub_vec = wl_vec[pix_vec]
    
    # Minimum and maximum wavelengths
    wl_min = np.min(wl_vec)
    wl_max = np.max(wl_vec)
    wl_ave = np.average(wl_vec)

    # Which wavelengths are associated with the pixel combinations?
    wl_v0 = wl_vec[cmb_pix[:,0]]
    wl_v1 = wl_vec[cmb_pix[:,1]] 


    # Create the [lambda_min,lambda_max] pairs that delimit a "bin"
    wl_binm = wl_vec[bins]
    wl_binM = wl_vec[bins[1::]]
    wl_binM = np.append(wl_binM,wl_vec[-1])
    
    ### Calculate the temperature with the simple model
    Tave,std,rse,logR = ce_temperature(data_spl,wl_v0,wl_v1)
    print("Simple temperature model:",Tave,std,rse) 
    sol = None
    
    ### Do we have a "good enough" fit?   
    # If not, we assume first a linear function of emissivity and iterate
    # from there
    nunk = 2
    perturb = False
    sol_all = []
    while rse > rse_threshold and nunk < max_poly_order:
        refined_fit = True

        # What is our previous standard error?
        rse_nm1 = rse
        
        # Define the goal function
        f = lambda pc: goal_function(pc,logR,wl_v0,wl_v1,wl_min,wl_max)
#        filtered_data = splev(wl_sub_vec,data_spl)
#        bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
#        f = lambda pc: mixed_goal_function(pc,logR,
#                                           wl_v0,wl_v1,
#                                           wl_min,wl_max,
#                                           wl_sub_vec,filtered_data,
#                                           bb_eps)
        
        # Initial values of coefficients
        pc0 = np.zeros(nunk)
        pc0[0] =  0.5     
        
        if perturb:
            pc0[0] = 0
            while pc0[0] == 0:
                pc0[0] = np.random.sample()
               

        # Minimization
        min_options = {'xatol':1e-15,'fatol':1e-15,'maxfev':5000} # Nelder-Mead
#        min_options = {'gtol':1e-15} # BFGS
#        min_options = {'tol':1e-15,'maxiter':20000} # COBYLA
#        min_options = {'ftol':1e-15,'eps':1e-2} # SLSQP
        sol = minimize(f,pc0,method='Nelder-Mead',options=min_options)
#        sol = basinhopping(f,pc0)

        # Calculate temperature from solution
        Tave,std,rse = nce_temperature(sol.x,logR,
                    wl_v0,wl_v1,
                    wl_binm,wl_binM,
                    wl_min,
                    wl_max)
        
        print("Advanced temperature model:",Tave,std,rse,sol.x,pc0[0])
        
#        # If our new standard error is BIGGER than the previous one
#        # AND we did NOT perturb the initial values already
#        # THEN we decrease the order of the polynomial and perturb
#        if rse_nm1 < rse and not perturb:
#            nunk = nunk-1
#            perturb = True
#            if nunk == 1:
#                nunk = 2
#        # If we already perturbed the vector and it did not go well, we just
#        # go ahead and perturb it again
#        elif rse_nm1 < rse and perturb:
#            perturb = True
#        else:
#            nunk = nunk + 1
#            perturb = False
        nunk = nunk + 1
        sol_all.append(sol.x)
    
    return Tave,std,rse,refined_fit,sol,sol_all
   
'''
Function: compute_poly_temperature
Calculates the temperature based on the assumption of a polynomial order
Inputs:
    - data_spl Spline representation of the filtered intensity data
    - cmb_pix Pixels chosen for each pixel bin
    - pix_vec Overall pixel vector
    - wl_vec Vector of wavelengths (nm)
Ouputs:
    - Predicted temperature from averaging (K)
    - Standard deviation (K)
    - Standard deviation (%)
    - Flag indicating if advanced method was used
'''
def compute_poly_temperature(data_spl,cmb_pix,pix_vec,wl_vec,order):    
    bins = pix_vec[0::pix_slice]
    wl_sub_vec = wl_vec[pix_vec]
    
    # Minimum and maximum wavelengths
    wl_min = np.min(wl_sub_vec)
    wl_max = np.max(wl_sub_vec)
    wl_ave = np.average(wl_vec)

    # Which wavelengths are associated with the pixel combinations?
    wl_v0 = wl_vec[cmb_pix[:,0]]
    wl_v1 = wl_vec[cmb_pix[:,1]] 


    # Create the [lambda_min,lambda_max] pairs that delimit a "bin"
    wl_binm = wl_vec[bins]
    wl_binM = wl_vec[bins[1::]]
    wl_binM = np.append(wl_binM,wl_vec[-1])
    
    ### Calculate the temperature with the simple model
    Tave,std,rse,logR = ce_temperature(data_spl,wl_v0,wl_v1)
    sol = None
    
    # Define the goal function
    filtered_data = splev(wl_sub_vec,data_spl)
    bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
    f = lambda pc: mixed_goal_function(pc,logR,
                                       wl_v0,wl_v1,
                                       wl_min,wl_max,
                                       wl_sub_vec,filtered_data,
                                       bb_eps)
#    f = lambda pc: goal_function(pc,logR,wl_v0,wl_v1,wl_min,wl_max)
    
    # Initial values of coefficients
    pc0 = np.zeros(order+1)
    pc0[0] =  1     

    # Minimization
    min_options = {'xatol':1e-15,'fatol':1e-15,'maxfev':5000} # Nelder-Mead
    sol = minimize(f,pc0,method='Nelder-Mead',options=min_options)
#    def fconst(coeffs,wl_sub_vec,wl_min,wl_max):
#        cheb = Chebyshev(coeffs,[wl_min,wl_max])
#        val = chebyshev.chebval(wl_sub_vec,cheb.coef)       
#        sgn = np.sign(val)
#        sgn = np.prod(sgn)
#        return sgn
    
#    cons = ({'type': 'ineq', 'fun': lambda coeff: fconst(coeff,wl_sub_vec,wl_min,wl_max) })
#    min_options = {'tol':1e-15,'maxiter':5000,'rhobeg':100} # COBYLA
#    sol = minimize(f,pc0,method='COBYLA',constraints=cons,options=min_options)
    
#    sol = basinhopping(f,pc0)
    
    # Calculate temperature from solution
    Tave,std,rse = nce_temperature(sol.x,logR,
                        wl_v0,wl_v1,
                        wl_binm,wl_binM,
                        wl_min,
                        wl_max)
    
    return Tave,std,rse,sol    


'''
Function: nce_temperature
Calculates the temperature based on a Non-Constant Emissivity (NCE).
The emissivity is modeled with a Chebyshev polynomial of order N where N 
is determined by a separate routine
Inputs:
    - 
Outputs:
    - Predicted temperature from averaging (K)
    - Standard deviation (K)
    - Standard deviation (%)
 '''
def nce_temperature(poly_coeff,logR,
                    wl_v0,wl_v1,
                    wl_binm,wl_binM,
                    wl_min,
                    wl_max):  
    
    ### Polynomial representation of the emissivity
    poly_eps = Polynomial(poly_coeff,[wl_min,wl_max])
    
    ### Emissivities at the wavelengths of interest
    eps1 = polynomial.polyval(wl_v1,poly_eps.coef)
    eps0 = polynomial.polyval(wl_v0,poly_eps.coef)
    
    ### Inverse temperature
    try:
        invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
        
        ### Temperature
        Tout = 1/invT
        Tout *= C2 * ( 1/wl_v1 - 1/wl_v0)
    
        ### Returns
        Tave,std,rse,_ = tukey_fence(Tout)
    except:
        Tave,std,rse = 1e5 * np.ones(3)
    
    return Tave,std,rse    

# Calibration would be as follows
# Use a blackbody to calibrate instrument such that a CONSTANT value is used
# as a scaling factor:
# V_bb = A / l^5 exp(-C2/l*T_cal)
# A contains detector characteristics and also C1
# Assume that "A" does not change with temperature:
# V_measured = A * eps / l^5 exp(-C2/l*T_true)
# V_measured / V_bb = eps * exp(-C2/l*T_true) * exp(C2/l*T_cal)

def target_alt(log_eps,args):    
    wl_sub_vec, logR, Tcal = args
    
    Tvec = C2 / wl_sub_vec
    Tvec /= log_eps - logR + C2/(wl_sub_vec * Tcal)
    
#    print(np.mean(Tvec),np.std(Tvec))
    
    return np.std(Tvec)


def compute_temperature_alt(Tguess,Tcal,V_bb,V_meas,
                            chosen_pix, pix_vec,pix_sub_vec,
                            wl_vec,wl_sub_vec):
        
    logR = V_meas - V_bb[chosen_pix]
    logR /= np.log(10)
    
    wl_extract = wl_vec[chosen_pix]
    
    # First pass: determine log_eps0
    A = np.eye(len(chosen_pix))
#    np1_vec = -C2 / wl_extract * np.ones(len(chosen_pix))
#    np1_vec = np.array([np1_vec])
#    A = np.concatenate((A,np1_vec.T),axis=1)
    b = logR - C2/(wl_extract * Tcal) + C2/(wl_extract * Tguess)
    
#    sol_lsq = lsq_linear(A,b)
#    sol_lsq = np.linalg.inv(np.matmul(A.T,A))
#    sol_lsq = np.matmul(sol_lsq,A.T)
#    sol_lsq = np.matmul(sol_lsq,b)
    
#    sol_lsq = np.matmul(np.linalg.pinv(A),b)
#    sol_lsq = np.matmul(np.linalg.inv(A),b)
    
#    print(sol_lsq.x,1/sol_lsq.x[-1])
    
#    log_eps0 = 0.5 * np.ones(len(chosen_pix))
#    log_eps0 = np.log(log_eps0)
#    log_eps0 = np.copy(sol_lsq)
    log_eps0 = np.copy(b)
    sol_lsq = np.copy(b)
    
    args = [wl_vec[chosen_pix] , logR, Tcal]
    
#    min_options = {'xatol':1e-15,'fatol':1e-15,'maxfev':5000}
    min_options = {'gtol':1e-15,'ftol':1e-15,'maxiter':5000}
    
    bounds = []
    for idx in chosen_pix:
        bounds.append((-1e5,0))
    
    sol = minimize(target_alt,x0 = log_eps0, args = args, 
                   method='TNC', bounds=bounds,options=min_options)

    


#    print(sol)
    Tout = C2 / wl_vec[chosen_pix]
    Tout /= sol.x - logR + C2/(wl_vec[chosen_pix] * Tcal)
    std_Tout = np.std(Tout)
    Tout = np.mean(Tout)
    

    print(Tout,std_Tout)

    return sol,sol_lsq,Tout

