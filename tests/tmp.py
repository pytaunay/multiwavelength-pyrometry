### Numerical tests with PDFs

from scipy.stats import norm,uniform
import numpy
import matplotlib.pyplot as plt

Nelem = 10000

### Default from normal distribution
xdefault = norm.rvs(0,1,Nelem)
xdisplay = norm.rvs(0,1,2*Nelem) # Display with same # of elements

### Offset from some other distribution
delta = uniform.rvs(0,3,Nelem)

### With offset
xoff = xdefault + delta

plt.hist(xoff,100)
plt.hist(xdisplay,100)