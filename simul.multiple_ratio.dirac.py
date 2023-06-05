#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

memory = 1



f = TimeFunction(0,1,1000, 1)
f.TFdirac(0.25)


#ratio_slope = [1, 1.27, 1.49, 1.86, 2.29, 2.80, 3.67, 4.83, 6.15, 7.89, 9.39, 11.26, 13.59, 12.27, 15.01,
#1.04, 1.28, 1.52, 1.93, 2.39, 2.90, 3.65, 5.33, 6.65, 7.63, 9.84, 11.66, 13.87, 15.92]
#facteur = 5.27*10**-7

#ratio_slope = [5.0, 3.89, 3.52, 3.13, 2.44, 2.06, 1.86, 3.85, 3.66, 3.01, 2.47, 2.17,2.21, 3.12]
#ratio_slope = [2.1, 2.0, 1.93, 1.83, 1.73, 1.64, 1.55, 1.47, 1.38, 1.31, 1.23, 1.16, 1.09,
#        1.03, 0.96, 0.91]
#ratio_slope = [1.08, 1.14, 0.73, 0.98, 1.04, 1.03, 0.75, 0.70, 0.73, 0.69, 0.62, 0.60,
#        0.52, 0.55]

alpha=[0.5]
ratio_slope = [3.85, 3.47, 2.63, 2.75, 3.34, 3.09, 2.16, 2.64, 2.22, 2.08, 1.60, 1.83,
        1.78, 1.62]


facteur = 10**-6
ratio_pente = [facteur * i for i in ratio_slope]
kd= 13600
ratio_pente_kd = [kd * i for i in ratio_pente]
print(trng_entropy(alpha, f, memory, 14, ratio_pente_kd, True))
print(find_waiting_time(alpha, f, memory, 14, ratio_pente, [5000,10000], 0.997, 0.001,True))
