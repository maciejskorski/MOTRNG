#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
memory = 4



f = TimeFunction(0,1,1000, 1)
f.TFconst(1)
print f.TFsum()

#ratio_slope = [1, 1.27, 1.49, 1.86, 2.29, 2.80, 3.67, 4.83, 6.15, 7.89, 9.39, 11.26, 13.59, 12.27, 15.01,
#1.04, 1.28, 1.52, 1.93, 2.39, 2.90, 3.65, 5.33, 6.65, 7.63, 9.84, 11.66, 13.87, 15.92]
#facteur = 5.27*10**-7

ratio_slope = [5.0, 3.89, 3.52, 3.13, 2.44, 2.06, 1.86, 3.85, 3.66, 3.01, 2.47, 2.17,2.21, 3.12]
facteur = 10**-7
ratio_pente = [facteur * i for i in ratio_slope]

print find_waiting_time(alpha, f, memory, 14, ratio_pente, [8000, 15000], 0.997, 0.001, True)
