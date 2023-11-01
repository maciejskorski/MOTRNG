#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
memory = 0
precision = 1000
nxor = 4
sigmat = 2.5*10**-6
D = 2000

f = TimeFunction(0,1,precision, 1)
f.TFdirac(0.25)

alpha_list = [alpha]
ratio_slope_list = [sigmat]
ratio_quality_list = [0.05]

#ent,_ = trng_entropy(alpha_list, f, memory, nxor, ratio_quality_list, True)
#print(ent)

for i in [2, 4, 8, 16, 32, 64]:
    nxor =i
    D, _= find_waiting_time(alpha_list, f, memory, nxor, ratio_slope_list, [1, 40000],0.997, 0.000001, False)
    print(i,sigmat*D[0])

