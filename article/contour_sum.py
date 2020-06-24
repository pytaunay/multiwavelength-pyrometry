# MIT License
# 
# Copyright (c) 2020 Pierre-Yves Camille Regis Taunay
#  
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
File: contour_sum.py

Description: calculate the sum "S" and corresponding asymptotic expansions.
The sum is calculated under the assumption of linearly spaced wavelengths.

This file can be used to generate Figures 1 through 3 in our 2020 RSI Journal
article.
'''

import numpy as np

from scipy.special import polygamma as PolyGamma
from scipy.optimize import minimize

from numpy import euler_gamma as EulerGamma


import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import matplotlib.colors
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec


### Total sum as calculated with Mathematica
totalsum = lambda N,R: (1/360)*1/(N**2*R**2)*(12*EulerGamma*((-30)+(-60)*R+(-30) 
  *((-1)+N)**(-2.)*(1+3*((-1)+N)*N)*R**2+30*(1+(-2)*N)*(( 
  -1)+N)**(-2.)*N*R**3+((-1)+N)**(-4.)*(1+(-15)*((-1)+N)**2* 
  N**2)*R**4)+N*(3*((-1)+N)**(-2.)*N*R**2*(120+R*(120+(( 
  -1)+N)**(-2.)*(19+N*((-62)+39*N))*R))+2*np.pi**2*(30+R*(60+( 
  (-1)+N)**(-1.)*R*((-30)+60*N+30*N*R+((-1)+N)**(-2.)*(1+N+ 
  (-9)*N**2+6*N**3)*R**2))))+(-12)*(30+60*R+30*((-1)+N)**( 
  -2)*(1+3*((-1)+N)*N)*R**2+30*((-1)+N)**(-2.)*N*((-1)+2* 
  N)*R**3+((-1)+N)**(-4.)*((-1)+15*((-1)+N)**2*N**2)*R**4)* 
  PolyGamma(0,1+N)+(-12)*N*(30+R*(60+((-1)+N)**(-1.)*R*((-30) 
  +60*N+30*N*R+((-1)+N)**(-2.)*(1+N+(-9)*N**2+6*N**3)* 
  R**2)))*PolyGamma(1,1+N));
  
mean_term = lambda N,R: N*(N-1)*totalsum(N,R)


largeNasymptote = lambda N,R: np.pi**2/6 * 1/(N*R**2) * (1+2*R+2*R**2+R**3+R**4/5)




### PART 1 : Calculate the asymptotic error for four different number of 
### wavelengths
dlamvec = np.logspace(-3,1,1000)
Nvec = np.array([10,100,1000,10000],dtype=np.float64)
zerr = []

for N in Nvec:
    s = totalsum(N,dlamvec)
    la = largeNasymptote(N,dlamvec)
    
    err = np.abs(s-la)/s * 100

    zerr.append(err)

# Plot asymptotic expansion for four different N
plt.figure()
plt.ylim([0.01,100])

for idx,N in enumerate(zerr):
    plt.loglog(dlamvec,zerr[idx],'k--')


### PART 2 : Contour plot of the sum
plt.figure()
Rvec = np.logspace(-1,1,100)
Nvec = np.logspace(np.log10(2),3,100)

Rv,Nv = np.meshgrid(Rvec,Nvec)
allvec = totalsum(Nv,Rv)
meanvec = mean_term(Nv,Rv)

gs = gridspec.GridSpec(1, 2,width_ratios=[15,1])
ax1 = plt.subplot(gs[0])

levels = np.logspace(-3,1,30)
cs = ax1.contour(Rv, Nv, allvec,levels=levels,cmap=cm.gray,norm = LogNorm())

ax1.set_yscale('log')
ax1.set_xscale('log')


### PART 3 : Find the best R for each N
minall = []
Nvecint = np.logspace(np.log10(2),3,1000,dtype=np.int64)
Nvecint = np.unique(Nvecint)
Nvecint = np.array(Nvecint,dtype=np.float64)
minoptions = {'ftol':1e-15,'xtol':1e-15}

for N in Nvecint:
    sol = minimize(lambda R:mean_term(N,R),
                   1.45987,
                   method='Nelder-Mead',
                   options = minoptions
                   )
#    print(N,sol.x)
    minall.append(sol.x)

ax2 = plt.subplot(gs[1])
levels = np.linspace(0.001,0.01,5)
levels = np.concatenate((levels[:-1],np.linspace(0.01,0.1,5)))
levels = np.concatenate((levels[:-1],np.linspace(0.1,1,5)))
levels = np.concatenate((levels[:-1],np.linspace(1,10,5)))

plt.plot(Nvecint[1:],minall[1:],'k.')
plt.ylim([2,100])