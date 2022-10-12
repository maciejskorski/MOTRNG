#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.50
memory = 5
precision = 1000
nxor = 2
sigmat = 7.5*10**-6
D= 2000

f = TimeFunction(0,1,precision, 1)
f.TFconst(1)

alpha_list = [alpha]
ratio_slope_list = [sigmat]
ratio_quality_list = [sigmat*D]
#alpha_list = [alpha]*nxor
#ratio_slope_list = [sigmat*D]*nxor


ent,_ = trng_entropy(alpha_list, f, memory, nxor, ratio_quality_list, True)
print(ent)

print(find_waiting_time(alpha_list, f, memory, nxor, ratio_slope_list, [1000, 40000],0.997, 0.001, True))
