#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
memory = 8

f = TimeFunction(0,1,5000, 1)
f.TFconst(1)

ratio_slope = [1]
facteur = 10**-6
ratio_slope = [facteur * i for i in ratio_slope]

#print find_waiting_time([alpha], f, memory, 14, ratio_slope, [1000, 2000], 0.997, 0.001, True)
print trng_entropy([alpha], f, memory, 1, [0.1])
