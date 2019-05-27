import numpy as np
import spectropyrometer_constants as sc
import generate_spectrum as gs
import pixel_operations as po
import temperature_functions as tf

from numpy.polynomial import Polynomial, polynomial
from scipy.interpolate import splrep,splev
from scipy.stats import normaltest
from statistics import tukey_fence

### Artificial temperature and emissivity
Ttrue = 1500

art_wl = np.array([300,500,1100])
art_eps_data = np.array([1,0.3,1])
art_fac = np.polyfit(art_wl,art_eps_data,deg=2)

a0,a1,a2 = art_fac
art_eps = lambda wl,T: a0*wl**2 + a1*wl + a2

### Vectors of pixels and wavelengths
wl_vec = np.linspace(300,1100,(int)(3000))
pix_vec = np.linspace(0,2999,3000)
pix_vec = np.array(pix_vec,dtype=np.int64)


I_calc,noisy_data,filtered_data,data_spl,pix_sub_vec = gs.generate_data(wl_vec,Ttrue,pix_vec,art_eps)

pix_sub_vec = pix_vec[sc.window_length:-sc.window_length]
wl_sub_vec = wl_vec[pix_sub_vec]

chosen_pix = po.choose_pixels(pix_sub_vec,bin_method='average')
cmb_pix = po.generate_combinations(chosen_pix,pix_sub_vec)


### Pixel operations
bins = pix_sub_vec[0::sc.pix_slice]

# Minimum and maximum wavelengths
wl_min = np.min(wl_sub_vec)
wl_max = np.max(wl_sub_vec)

# Which wavelengths are associated with the pixel combinations?
wl_v0 = wl_vec[cmb_pix[:,0]]
wl_v1 = wl_vec[cmb_pix[:,1]] 

# Create the [lambda_min,lambda_max] pairs that delimit a "bin"
wl_binm = wl_vec[bins]
wl_binM = wl_vec[bins[1::]]
wl_binM = np.append(wl_binM, wl_vec[-1])

### Apply tests
logR = tf.calculate_logR(data_spl, wl_v0, wl_v1)
domain = np.array([wl_min,wl_max])
## Different cases of temperature distributions
# Case 1: _exact_ emissivity
pol =  Polynomial(np.array([a2,a1,a0]),domain)
eps1 = polynomial.polyval(wl_v1,pol.coef)
eps0 = polynomial.polyval(wl_v0,pol.coef)
invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
Tout = 1/invT
Tout *= sc.C2 * ( 1/wl_v1 - 1/wl_v0)

T_exact = np.copy(Tout)
print(normaltest(T_exact))

# Case 2: similar emissivity, but not exact
a0_star = 8.55164533e+00 
a1_star = -2.38980909e-02  
a2_star = 1.70904144e-05
art_fac_similar = np.array([a2_star,a1_star,a0_star])
      

pol =  Polynomial(art_fac_similar[::-1],domain)
eps1_similar = polynomial.polyval(wl_v1,pol.coef)
eps0_similar = polynomial.polyval(wl_v0,pol.coef)

invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0_similar/eps1_similar)
Tout = 1/invT
Tout *= sc.C2 * ( 1/wl_v1 - 1/wl_v0)
Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout, method = 'dispersion')

T_similar = np.copy(Tleft)
print(normaltest(Tleft))


# Case 3: dissimilar emissivity, same order
art_eps_invert = np.array([0.3,0.8,0.3])
art_fac = np.polyfit(art_wl,art_eps_invert,deg=2)

a0_inv,a1_inv,a2_inv = art_fac

pol =  Polynomial(np.array([a2_inv,a1_inv,a0_inv]),domain)
eps1 = polynomial.polyval(wl_v1,pol.coef)
eps0 = polynomial.polyval(wl_v0,pol.coef)

invT = logR - 5 *np.log(wl_v1/wl_v0) - np.log(eps0/eps1)
Tout = 1/invT
Tout *= sc.C2 * ( 1/wl_v1 - 1/wl_v0)


Tave, Tstd, Tmetric, Tleft = tukey_fence(Tout, method = 'dispersion')
T_inv = np.copy(Tleft)

print(normaltest(T_inv))