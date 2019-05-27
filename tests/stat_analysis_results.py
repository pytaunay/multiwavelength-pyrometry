#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:34:29 2019

@author: pyt
"""
import numpy as np
import matplotlib.pyplot as plt

Nwl = np.array([580,290,116,58,29,15,10])
Rp1 = np.array([2.4646,2.4604,2.4476,2.4263,2.384,2.3837,2.299])

#### DIRECTLY FROM SIMULATIONS
#### 1000 MC samples, T = 1500 K, sigma = 0.01, 
#### dlambda = 5, 10, 25, 50, 100, 200, 300
mu_mc = np.array([-3.45e-4,
                  -3.5e-4,
                  -3.8e-4,
                  -3.96e-4,
                  -4.47e-4,
                  -5.05e-4,
                  -5.72e-4])

sig_mc = np.array([1.08e-4,
                   1.02e-4,
                   1.065e-4,
                   1.04e-4,
                   1.39e-4,
                   1.81e-4,
                   2.19e-4])

### APPROXIMATION
mu_app = np.array([2.027e-4,
                   9.98e-5,
                   3.84e-5,
                   1.81e-5,
                   8.11e-6,
                   3.65e-6,
                   2.01e-6])

sig_app = np.array([3.47e-5,
                    4.88e-5,
                    7.58e-5,
                    1.045e-4,
                    1.414e-4,
                    1.86e-4,
                    2.12e-4])

### ACCURATE EXPANSION
mu_acc = np.array([-1.35e-4,
                   -2.89e-4,
                   -3.64e-4,
                   -3.997e-4,
                   -4.42e-4,
                   -4.98e-4,
                   -5.75e-4])

sig_acc = np.array([3.67e-5,
                    4.95e-5,
                    7.6e-5,
                    1.046e-4,
                    1.414e-4,
                    1.86e-4,
                    2.12e-4])


fig,ax = plt.subplots(1,1)
ax.plot(Nwl,sig_mc,'k^')
ax.plot(Nwl,sig_app,'^')
ax.plot(Nwl,sig_acc,'ko')

#ax[1].plot(Nwl,mu_mc,'k^')
##ax[1].plot(Nwl,mu_app,'^')
#ax[1].plot(Nwl,mu_acc,'ko')

C2 = 14384000
T0 = 1500
lambda_0 = 300
#R = np.average(Rp1)
R = 7

#Nlim = (C2/(T0*lambda_0) * R)
Nlim = C2/(T0*lambda_0) * R/(1+R)**2*0.1/0.01
ax.axvline(x=Nlim,color='k',linestyle='--')
#ax[1].axvline(x=Nlim,color='k',linestyle='--')

#### DIRECTLY FROM SIMULATIONS
#### 1000 MC samples, T = 3000 K, sigma = 0.01, 
#### dlambda = 5, 10, 25, 50, 100, 200, 300
#mu_mc = np.array([-5.6e-5,
#                  -6.74e-5,
#                  -7.70e-5,
#                  -9.54e-5,
#                  -1.079e-4,
#                  -1.069e-4,
#                  -1.127e-4])
#
#sig_mc = np.array([1.09e-4,
#                   1.073e-4,
#                   1.08e-4,
#                   1.05e-4,
#                   1.466e-4,
#                   1.964e-4,
#                   2.189e-4])
#
#### APPROXIMATION
#mu_app = np.array([2.10e-4,
#                   1.036e-4,
#                   4.01e-5,
#                   1.91e-5,
#                   8.77e-6,
#                   3.98e-6,
#                   2.288e-6])
#
#sig_app = np.array([3.53e-5,
#                    4.973e-5,
#                    7.76e-5,
#                    1.08e-4,
#                    1.470e-4,
#                    1.946e-4,
#                    2.25e-4])
#
#### ACCURATE EXPANSION
#mu_acc = np.array([1.56e-4,
#                   2.802e-6,
#                   -6.57e-5,
#                   -8.85e-5,
#                   -1.02e-4,
#                   -1.1e-4,
#                   -1.18e-4])
#
#sig_acc = np.array([3.7e-5,
#                    5.03e-5,
#                    7.77e-5,
#                    1.076e-4,
#                    1.470e-4,
#                    1.946e-4,
#                    2.25e-4])


#### DIRECTLY FROM SIMULATIONS
#### 1000 MC samples, T = 3000 K, sigma = 0.1, 
#### dlambda = 5, 10, 25, 50, 100, 200, 300
#mu_mc = np.array([3.3e-3,
#                  3.834e-3,
#                  3.38e-3,
#                  2.366e-3,
#                  8.69e-4,
#                  1.822e-4,
#                  2.09e-4])
#
#sig_mc = np.array([5.759e-3,
#                   5.09e-3,
#                   3.832e-3,
#                   1.789e-3,
#                   1.504e-3,
#                   2.03e-3,
#                   2.33e-3])
#
#### APPROXIMATION
#mu_app = np.array([2.09e-2,
#                   0.0104,
#                   4.01e-3,
#                   1.91e-3,
#                   8.80e-4,
#                   3.98e-4,
#                   2.29e-4])
#
#sig_app = np.array([3.35e-4,
#                    4.9729e-4,
#                    7.757e-4,
#                    1.08e-3,
#                    1.470e-3,
#                    1.946e-3,
#                    2.255e-3])
#
#### ACCURATE EXPANSION
#mu_acc = np.array([2.679e-4,
#                   1.22e-3,
#                   3.43e-3,
#                   2.27e-3,
#                   8.02e-4,
#                   2.874e-4,
#                   1.09e-4])
#
#sig_acc = np.array([1.16e-3,
#                    9.26e-4,
#                    9.16e-4,
#                    1.13e-3,
#                    1.49e-3,
#                    1.952e-3,
#                    2.26e-3])