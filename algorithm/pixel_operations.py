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

import numpy as np
import itertools

from algorithm.spectropyrometer_constants import pix_slice

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
        if pix_slice > 1:
            chosen_pix = pix_vec[0::(int)(lpix_slice/2)][1::2]
        else:
            chosen_pix = np.copy(pix_vec)
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


