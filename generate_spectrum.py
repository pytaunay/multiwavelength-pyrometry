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

from spectropyrometer_constants import C1,C2

'''
Function: wien_approximation
Calculates the Wien approximation to Planck's law for non-constant emissivity
Inputs:
    - lnm: wavelength in nm
    - T: temperature in K
    - f_eps: a lambda function representing the emissivity as function of
    temperature and wavelength
'''
def wien_approximation(lnm,T,f_eps):    
    eps = f_eps(lnm,T) # Emissivity
    
    return eps * C1 / lnm**5 * np.exp(-C2/(T*lnm))