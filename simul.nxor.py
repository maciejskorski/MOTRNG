#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
memory = 5
precision = 1000
nxor = 5
sigmat = 2*10**-6
D= 2000

f = TimeFunction(0,1,precision, 1)
f.TFconst(1)
print f.TFsum()

alpha_list = [alpha]
ratio_slope_list = [sigmat*D]
#alpha_list = [alpha]*nxor
#ratio_slope_list = [sigmat*D]*nxor


ent,_ = trng_entropy(alpha_list, f, memory, nxor, ratio_slope_list, True)
print(ent)

