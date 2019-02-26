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

from spectropyrometer_constants import pix_slice

''' Function: choose_pixels
Finds the pixels from which we will compute the "two-wavelengths" temperatures
Inputs:
    - pix_vec: vector of pixels with which we work
    - bin_method: the method by which a pixel is picked from a bin. "Average"
    uses the average value of the boundaries of the pixel bin. "Random" picks
    a random pixel within the boundaries of the pixel bin
Outputs:
    - chosen_pix: vector of chosen pixels
'''
def choose_pixels(pix_vec,bin_method='average'):
    ### Bin the pixels
    bins = pix_vec[0::pix_slice]
    
    ### For each bin, find a pixel
    chosen_pix = []
    for idx in bins[:-1]:
        # Find the low and high bounds of that slice
        lo = pix_vec[idx]
        hi = pix_vec[idx + pix_slice - 1]
        
        # Pick a pixel in the bin
        # Can be the average of the boundaries of the bin or a random pixel
        if bin_method == 'average':
            pix_idx = (int)(np.average([lo,hi]))
        elif bin_method == 'random':
            pix_idx = np.random.choice(pix_vec[lo:hi],size=1)[0]
        
        
        chosen_pix.append(pix_idx)
    chosen_pix = np.array(chosen_pix)  

    return chosen_pix


'''
Function: generate_combinations
Generates the pixel combinations that are used to compute the estimated 
"two-wavelengths" temperature
Inputs:
    - chosen_pix: the subset of pixels with which we work 
    - pix_vec: all of the pixels
Outputs:
    - cmb_pix: the array of pixel combinations    
'''
def generate_combinations(chosen_pix,pix_vec):
    cmb_pix = []

    # For each pixel p0 we picked...
    for p0 in chosen_pix:
        
        # Get corresponding pair pixel above and below this pixel p0
        # They belong to other slices
        p1vec_p = pix_vec[p0::pix_slice]
        p1vec_m = pix_vec[p0::-pix_slice]
        
        # Create a vector of pixels, remove any duplicates, make sure we do not
        # include p0
        p1vec = np.concatenate((p1vec_m,p1vec_p))
        p1vec = np.unique(p1vec)
        p1vec = p1vec[p1vec != p0]
        
        # Create the list of combinations        
        for p1 in p1vec:      
            cmb_pix.append((p0,p1))
            
    cmb_pix = np.array(cmb_pix)
    return cmb_pix


