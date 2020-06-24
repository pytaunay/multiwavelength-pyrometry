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

### Define some constants
# Blackbody curve
C1 = 1.191e16 # W/nm4/cm2 Sr
C2 = 1.4384e7 # nm K

### Constants relevant to the algorithm
max_poly_order = 4 # Maximum polynomial order to consider
threshold = 1e-3 # Threshold before the fit is considered to be "bad"
eps0 = 0.5 # Initial value for the emissivity coeff.
ksplits = 10 # Number of k folds

### Pixel and window length
## Window length
window_length = 51 # Window length for the moving average
#window_length = (int)(pix_slice+1) # alternative 

## Pixel slice
lchosen = 58 # Number of pixels chosen
# Total number of pixels to take into a slice
pix_slice = (int)((3000 - 2*(window_length-1))/lchosen) 
# Below are two alternatives that can be used if the total number of pixels
# available is small 
#pix_slice = 7
#pix_slice = 1

