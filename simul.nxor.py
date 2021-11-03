#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
memory = 5
precision = 1000
nxor = 2
sigmat = 2*10**-6
D= 2000

f = TimeFunction(0,1,precision, 1)
f.TFconst(1)
print f.TFsum()

ratio_slope = [sigmat*D]

ent,_ = trng_entropy([alpha], f, memory, nxor, ratio_slope, True)
print(ent)

