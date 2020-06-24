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

### Define some constants
# Blackbody curve
C1 = 1.191e16 # W/nm4/cm2 Sr
C2 = 1.4384e7 # nm K

### Constants relevant to the algorithm
max_poly_order = 8 # Maximum polynomial order to consider
threshold = 1e-8 # Threshold before the fit is considered to be "bad"
eps0 = 0.5 # Initial value for the emissivity coeff.
ksplits = 10 # Number of k folds

### Pixel and window length
## Pixel slice
lchosen = 58 # Number of pixels chosen
# Total number of pixels to take into a slice
pix_slice = (int)((3000 - 2*(window_length-1))/lchosen) 
# Below are two alternatives that can be used if the total number of pixels
# available is small 
#pix_slice = 7
#pix_slice = 1

## Window length
window_length = 51 # Window length for the moving average
#window_length = (int)(pix_slice+1) # alternative 
