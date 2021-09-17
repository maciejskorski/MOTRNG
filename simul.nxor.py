#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
memory = 10
precision = 1000
nxor = 5

#minq = 0.01
#inc = 0.01

f = TimeFunction(0,1,precision, 1)
f.TFconst(1)
print f.TFsum()

ratio_slope = [8000]*nxor
facteur = 10**-6
ratio_slope = [facteur * i for i in ratio_slope]

#for  i in range(10):
#    quality = minq + i*inc
#    quality_list = [quality]*nxor
ent,_ = trng_entropy([alpha], f, memory, nxor, ratio_slope, False)
print(ent)

#print find_waiting_time([alpha], f, memory, nxor, ratio_slope, [20000, 30000], 0.997, 0.001, False)
