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
import itertools

from spectropyrometer_constants import pix_slice


def choose_pixels(pix_vec,bin_method='average'):
    ''' 
    Finds the pixels from which we will compute the "two-wavelengths" tempera-
    tures
    Inputs:
        - pix_vec: vector of pixels with which we work
        - bin_method: the method by which a pixel is picked from a bin. "Average"
        uses the average value of the boundaries of the pixel bin. "Random" picks
        a random pixel within the boundaries of the pixel bin. "Median" picks
        the median value of the bin.
    Outputs:
        - chosen_pix: vector of chosen pixels
    '''
    ### In the case of the median it is easy to figure out the indexing
    if bin_method == 'median':
        # If the pixel slice is even, then we will have to add one to it
        if np.mod(pix_slice,2) == 0:
            lpix_slice = pix_slice + 1
        else:
            lpix_slice = pix_slice
        
        # Pick the median of each bin of size pix_slice by grabbing numbers
        # at every pix_slice / 2: [0::(int)(lpix_slice/2)]
        # Then grab only the center of each bin; otherwise we also grab the
        # beginning of each bin as well: [1::2]
        chosen_pix = pix_vec[0::(int)(lpix_slice/2)][1::2]
    else:
        ### Bin the pixels
        lo_bins = pix_vec[0::pix_slice]
        hi_bins = pix_vec[pix_slice-1::pix_slice]
        
        
        ### For each bin, find a pixel
        chosen_pix = []
        for idx in range(len(lo_bins)):
            # Find the low and high bounds of that slice
            lo = lo_bins[idx]
            
            # If the slice is out of bounds, pick last element
            try:
                hi = hi_bins[idx]
            except:
                hi = pix_vec[-1]
                    
            
            # Pick a pixel in the bin
            # Can be the average of the boundaries of the bin or a random pixel
            if bin_method == 'average':
                pix_idx = (int)(np.average([lo,hi]))
            elif bin_method == 'random':
                pix_idx = np.random.choice(pix_vec[lo:hi],size=1)[0]
            
            
            chosen_pix.append(pix_idx)
            
    chosen_pix = np.array(chosen_pix,dtype=np.int64)  

    return chosen_pix

def generate_combinations(chosen_pix,pix_vec):
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
    cmb_pix = []

    for i,j in itertools.combinations(chosen_pix,2):
        cmb_pix.append([i,j])
#            
    cmb_pix = np.array(cmb_pix)
    return cmb_pix


