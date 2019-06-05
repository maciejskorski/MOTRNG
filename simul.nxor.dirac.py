#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
memory = 1

f = TimeFunction(0,1,5000, 1)
f.TFdirac(0.25)
print f.TFsum()

ratio_slope = [1]
facteur = 5.27/2.0*10**-7
ratio_slope = [facteur * i for i in ratio_slope]
ent = trng_entropy(alpha, f, memory, 16, [0.025], True)

print "entropy"
print ent
print find_waiting_time(alpha, f, memory, 14, ratio_slope, [30000, 120000], 0.997, 0.001, True)
