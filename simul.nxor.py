#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

#alpha=0.50
memory = 5
precision = 10000
nxor = 55
#sigmat = 7.5*10**-6
D= 2000

f = TimeFunction(0,1,precision, 1)
f.TFconst(1)

alpha_list = [0.45]
#ratio_slope_list = [sigmat]
#ratio_quality_list = [sigmat*D]
ratio_slope_list = [3.2, 3.0, 2.6, 2.4, 2.2, 2.1, 1.6, 1.6, 1.3, 1.2, 1.1, 1.0, 0.88, 3.9,
        3.3, 3.1, 2.7, 2.5, 2.3, 2.1, 1.5, 1.5, 1.3, 1.2, 1.1, 1.0, 0.87, 3.7, 3.2, 2.9,
        2.6, 2.5, 2.2, 2.1, 1.7, 1.6, 1.4, 1.3, 1.1, 1.1, 0.82, 3.8, 3.2, 3.0, 2.5, 2.4,
        2.2, 2.0, 1.6, 1.5, 1.3, 1.2, 1.1, 1.0, 0.88]
ratio_slope_list = [x * 10**-7 for x in ratio_slope_list]
#alpha_list = [alpha]*nxor
#ratio_slope_list = [sigmat*D]*nxor


#ent,_ = trng_entropy(alpha_list, f, memory, nxor, ratio_quality_list, True)
#print(ent)

print(find_waiting_time(alpha_list, f, memory, nxor, ratio_slope_list, [1000, 40000],0.997, 0.001, True))
