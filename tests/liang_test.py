from scipy.optimize import minimize, lsq_linear

chosen_eps = art_eps

### Create a blackbody spectrum "calibration" curve
bb_eps = lambda wl,T: 1.0 * np.ones(len(wl))
l_cal = wl_sub_vec[0::400]
T_cal = 1500
I_bb = gs.wien_approximation(l_cal,T_cal,bb_eps)

### Generate some data
T = 1800
I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(
        wl_vec,T,pix_vec,chosen_eps)


I_val = 10**(filtered_data[0::400])


### Find initial value
Dvec = sc.C2/ (l_cal * T_cal) - np.log(I_val/I_bb)

yvec = np.copy(Dvec)
yvec *= -1

Amat = np.eye(len(l_cal))
np1_vec = -sc.C2 / l_cal
np1_vec = np.array([np1_vec])
Amat = np.concatenate((Amat,np1_vec.T),axis=1)

sol_lsq = lsq_linear(Amat,yvec)
print("Least-squre temp:", 1/sol_lsq.x[-1])

### Find minized problem
pair = (-1e6,0)
bounds = []
for it in range(len(l_cal)):
    bounds.append(pair)

def ftarget(X,l_vec,Dvec):
    
    tmp = sc.C2/l_vec * 1/(X+Dvec)
    print(tmp)
    return np.var(tmp)




# Perturbations
#per_array = np.array([3,1,1.01,1.05,1.1,1.5,0.99,0.95,0.9,0.5,0.1,-1])
per_array = np.array([3,1,1.01,1.05,1.1,1.5,0.99,0.95,0.9,0.5,0.1,-1])
f,ax = plt.subplots(1,2)
ax[0].semilogy(l_cal,I_val,'o')
ax[1].plot(l_cal,chosen_eps(l_cal,1),'o')
ax[1].plot(l_cal,np.exp(sol_lsq.x[0:-1]))

for per_fac in per_array:
    bfgs_options = {'ftol':1e-16,'gtol':1e-16,'maxiter':10000}
    if per_fac > 0:
        sol = minimize(lambda X: ftarget(X,l_cal,Dvec), sol_lsq.x[0:-1]*per_fac,
                       method='L-BFGS-B',
                       bounds=bounds, options=bfgs_options)
    else:
        neps = len(sol_lsq.x[0:-1])
        sol = minimize(lambda X: ftarget(X,l_cal,Dvec), 0.5*np.ones(neps),
                       method='L-BFGS-B',
                       bounds=bounds, options=bfgs_options)        

    Tvec = sc.C2/l_cal*1/(sol.x+Dvec)
    Tprediction = np.average(Tvec)
    
    print(per_fac,Tprediction,np.std(Tvec),sol.success)
    
    
    eps_reconstructed = lambda wl,T: np.exp(sol.x)
    
    I_rec = gs.wien_approximation(l_cal,Tprediction,eps_reconstructed)
    
    
    ax[0].semilogy(l_cal,I_rec)
    
    if per_fac == -1:
        ax[1].plot(l_cal,eps_reconstructed(l_cal,1),'^')
    else:
        ax[1].plot(l_cal,eps_reconstructed(l_cal,1))

#plt.plot(l_cal,np.exp(sol_lsq.x[0:-1]))
#plt.plot(l_cal,chosen_eps(l_cal,1))
#plt.plot(l_cal,np.exp(sol.x))


