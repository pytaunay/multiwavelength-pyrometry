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

from scipy.stats import iqr


def tukey_fence(Tvec, delta=31.3):
    '''
    Function: tukey_fence
    Descritpion: Removes outliers using Tukey fencing
    Inputs:
        - Tvec: some vector
        - delta: a fencing value above/below the third/first quartile, 
        respectively. Values outside of [Q1 - delta * IQR, Q3 + delta*IQR] are
        discarded
    Outputs:
        - Average of vector w/o outliers
        - Standard deviation of vector w/o outliers
        - Standard error of vector w/o outliers (%)
        - Vector w/o outliers
    '''      
    ### Exclude data w/ Tukey fencing
    T_iqr = iqr(Tvec)
    T_qua = np.percentile(Tvec,[25,75])
    
    min_T = T_qua[0] - delta * T_iqr
    max_T = T_qua[1] + delta * T_iqr
    
    T_left = Tvec[(Tvec>min_T) & (Tvec<max_T)]
    
    ### Calculate standard deviation, average of the fenced data
    Tstd = np.std(T_left)
    Tave = np.mean(T_left)
    
    ### Calculate a metric: coefficient of quartile dispersion
    dispersion = (T_qua[1] - T_qua[0]) / (T_qua[1] + T_qua[0])
    metric = dispersion

    return Tave, Tstd, metric, T_left