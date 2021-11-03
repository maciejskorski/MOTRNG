#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
memory = 0
precision = 1000
nxor = 1
sigmaT = 3*10**-6
D = 2000

f = TimeFunction(0,1,precision, 1)
f.TFdirac(0.25)
print f.TFsum()

ratio_slope = [D]*nxor
ratio_slope = [facteur * i for i in ratio_slope]

ent,_ = trng_entropy([alpha], f, memory, nxor, ratio_slope, False)
print(ent)


#print("Find waiting time")
#print find_waiting_time([alpha], f, memory, nxor, ratio_slope, [20000, 50000], 0.997, 0.001, False)
