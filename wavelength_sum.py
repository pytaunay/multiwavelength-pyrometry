import numpy as np

from scipy.special import polygamma
import matplotlib.pyplot as plt


totalsum = lambda N,dlam: (1/360)*(12*np.euler_gamma*((-30)+(-60)*((-1)+N)*dlam+(-30)*(1+3* 
  ((-1)+N)*N)*dlam**2+(-30)*((-1)+N)*N*((-1)+2*N)*dlam**3+(1+( 
  -15)*((-1)+N)**2*N**2)*dlam**4)+N*(3*N*dlam**2*(120+120*(( 
  -1)+N)*dlam+(19+N*((-62)+39*N))*dlam**2)+2*np.pi**2*(30+((-1)+N) 
  *dlam*(60+dlam*((-30)+60*N+30*((-1)+N)*N*dlam+(1+N+(-9)*N**2+ 
  6*N**3)*dlam**2))))+(-12)*(30+60*((-1)+N)*dlam+30*(1+3*((-1)+ 
  N)*N)*dlam**2+30*((-1)+N)*N*((-1)+2*N)*dlam**3+((-1)+15*(( 
  -1)+N)**2*N**2)*dlam**4)*polygamma(0,1+N)+(-12)*N*(30+((-1)+ 
  N)*dlam*(60+dlam*((-30)+60*N+30*((-1)+N)*N*dlam+(1+N+(-9)* 
  N**2+6*N**3)*dlam**2)))*polygamma(1,1+N))
  
dlamvec = np.logspace(-5,1,100)
#Nvec = np.array([2,10,100,1000,10000])
Nvec = np.linspace(2,10000,1000)

zeroAsymptote = lambda N: -np.euler_gamma + (1/6)*N*np.pi**2-polygamma(0,1+N)+N*polygamma(1,1+N)

zerr = []

for N in Nvec:
#    s = totalsum(N,dlamvec)
    
    z = zeroAsymptote(N)
    err = np.abs(z -  (1/6)*N*np.pi**2)/z * 100
#    print(N,np.euler_gamma/z,(1/6)*N*np.pi**2/z,polygamma(0,1+N)/z,N*polygamma(1,1+N)/z,zeroAsymptote(N)/z)
    zerr.append(err)
    
#    plt.loglog(dlamvec,s)


plt.semilogx(Nvec,zerr)
