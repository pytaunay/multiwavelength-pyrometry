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


pix_slice = 100 # Total number of pixels to take into a slice
max_poly_order = 4 # Maximum polynomial order to consider
cv_threshold = 0.3 # Coefficient of variation (%) before we consider the fit 
# to be "bad"

# Window length for the moving average
window_length = (int)(pix_slice) 
#if window_length % 2 == 0:
#    window_length += 1