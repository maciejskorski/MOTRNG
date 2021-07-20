#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
memory = 4
precision = 1000

f = TimeFunction(0,1,precision, 1)
f.TFconst(1)

#print find_waiting_time([alpha], f, memory, 14, ratio_slope, [1000, 2000], 0.997, 0.001, True)
print trng_entropy([alpha], f, 4, 2, [0.0004],True)
