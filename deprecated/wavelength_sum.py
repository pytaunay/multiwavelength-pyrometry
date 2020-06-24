import numpy as np

from scipy.special import polygamma as PolyGamma
from numpy import euler_gamma as EulerGamma
import matplotlib.pyplot as plt

from scipy.optimize import minimize


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
#smallNasymptote = lambda N,R: (1+R)**2/(4*R**2)
#smallNasymptote = lambda N,R: (13 * (90 + R * (180 + R * (162 + R * (72 + 13 * R)))))/(5184 * R**2)
#smallNasymptote = lambda N,R: (3 + R * (3 + R))**2/(36 * R**2)


dlamvec = np.logspace(-3,1,1000)
Nvec = np.array([10,100,1000,10000],dtype=np.float64)

#Nvec = np.arange(2,10,1)
#Nvec = np.array(Nvec,dtype=np.float64)
#
##zeroAsymptote = lambda N: -np.euler_gamma + (1/6)*N*np.pi**2-PolyGamma(0,1+N)+N*PolyGamma(1,1+N)
#
zerr = []

for N in Nvec:
    s = totalsum(N,dlamvec)
    la = largeNasymptote(N,dlamvec)
#    sa = smallNasymptote(N,dlamvec)
    
    err = np.abs(s-la)/s * 100
#    print(N,np.max(err))
    
#    plt.loglog(dlamvec,err)
    
#    z = zeroAsymptote(N)
#    err = np.abs(z -  (1/6)*N*np.pi**2)/z * 100
##    print(N,np.euler_gamma/z,(1/6)*N*np.pi**2/z,polygamma(0,1+N)/z,N*polygamma(1,1+N)/z,zeroAsymptote(N)/z)
#    zerr.append(err)
#    
##    plt.loglog(dlamvec,s)
#
#plt.ylim([0.001,100])
#plt.semilogx(Nvec,zerr)

Rvec = np.logspace(-1,1,100)
Nvec = np.logspace(np.log10(2),3,100)

Rv,Nv = np.meshgrid(Rvec,Nvec)
allvec = totalsum(Nv,Rv)
meanvec = mean_term(Nv,Rv)

#plt.contourf(Rv,Nv,allvec,levels=np.logspace(-1,3))
from matplotlib import ticker, cm
import matplotlib.colors
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec


### PLOT THE VARIANCE TERM
gs = gridspec.GridSpec(1, 2,width_ratios=[15,1])
ax1 = plt.subplot(gs[0])

levels = np.logspace(-3,1,30)
cs = ax1.contour(Rv, Nv, allvec,levels=levels,cmap=cm.gray,norm = LogNorm())

ax1.set_yscale('log')
ax1.set_xscale('log')

### Find the best R for each N
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

#ax1.plot(minall,Nvecint,'k.')
ax1.set_ylim([2,100])
#ax1.set_xlim([0.1,10])
#
#XC = [np.zeros(len(levels)), np.ones(len(levels))]
#YC = [levels, levels]
#CM = ax2.contourf(XC,YC,YC, levels=levels, norm = LogNorm(),cmap=cm.gray)
#
## log y-scale
#ax2.set_yscale('log')  
## y-labels on the right
#ax2.yaxis.tick_right()
## no x-ticks
#ax2.set_xticks([])
#
#
#### PLOT THE MEAN TERM
#gs = gridspec.GridSpec(1, 2,width_ratios=[15,1])
#ax1 = plt.subplot(gs[0])
#
#levels = np.logspace(-3,1,30)
##cs = ax1.contour(Rv, Nv, meanvec,levels=levels,cmap=cm.gray,norm = LogNorm())
#cs = ax1.contourf(Rv,Nv,meanvec, levels=np.logspace(-3,3,30))
#
#ax1.set_yscale('log')
#ax1.set_xscale('log')
#
#### Find the best R for each N
#minall = []
#Nvecint = np.logspace(np.log10(2),3,1000,dtype=np.int64)
#Nvecint = np.unique(Nvecint)
#Nvecint = np.array(Nvecint,dtype=np.float64)
#minoptions = {'ftol':1e-15,'xtol':1e-15}
##for N in Nvecint:
##    sol = minimize(lambda R:totalsum(N,R),
##                   1.45987,
##                   method='Nelder-Mead',
##                   options = minoptions
##                   )
###    print(N,sol.x)
##    minall.append(sol.x)
#
#ax2 = plt.subplot(gs[1])
#levels = np.linspace(0.001,0.01,5)
#levels = np.concatenate((levels[:-1],np.linspace(0.01,0.1,5)))
#levels = np.concatenate((levels[:-1],np.linspace(0.1,1,5)))
#levels = np.concatenate((levels[:-1],np.linspace(1,10,5)))
#
##ax1.plot(minall,Nvecint,'k.')
#ax1.set_ylim([2,100])
#ax1.set_xlim([0.1,10])
#
#XC = [np.zeros(len(levels)), np.ones(len(levels))]
#YC = [levels, levels]
#CM = ax2.contourf(XC,YC,YC, levels=levels, norm = LogNorm(),cmap=cm.gray)
#
## log y-scale
#ax2.set_yscale('log')  
## y-labels on the right
#ax2.yaxis.tick_right()
## no x-ticks
#ax2.set_xticks([])




#
#fig, ax = plt.subplots()
##levels = np.array([-3,-2,-1,0,1,2,3,4,5,6],dtype=np.float64)
#levels = np.logspace(-3,1,50)
#
#
#
#cs = ax.contour(Rv, Nv, allvec,levels=levels,cmap=cm.gray,norm = LogNorm())
#ax.set_yscale('log')
#ax.set_xscale('log')
##ax.clabel(cs, inline=1, fontsize=10)
#ax.set_ylim([1,1000])
##plt.colorbar()
##cbar = fig.colorbar(cs)
#/
#ax.axvline(x=1.45897,linestyle='--',color='k')
#
#levels = np.linspace(0.001,0.01,5)
#levels = np.concatenate((levels[:-1],np.linspace(0.01,0.1,5)))
#levels = np.concatenate((levels[:-1],np.linspace(0.1,1,5)))
#levels = np.concatenate((levels[:-1],np.linspace(1,10,5)))
#
#XC = [np.zeros(len(levls)), np.ones(len(levls))]
#YC = [levls, levls]
#CM = ax2.contourf(XC,YC,YC, levels=levls, norm = LogNorm())
## log y-scale
#ax2.set_yscale('log')  
## y-labels on the right
#ax2.yaxis.tick_right()
## no x-ticks
#ax2.set_xticks([])
#
##norm= matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
## a previous version of this used
##norm= matplotlib.colors.Normalize(vmin=cs.vmin, vmax=cs.vmax)
## which does not work any more
#sm = plt.cm.ScalarMappable(norm=norm, cmap = cs.cmap)
#sm.set_array([])
#fig.colorbar(sm, ticks=np.array([1e-3,1e-2,1e-1,1,10]),spacing='proportional')
