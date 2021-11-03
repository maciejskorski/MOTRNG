#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
memory = 0
precision = 1000
variance = 7*10**-6
D= 2000
quality = D *variance

f = TimeFunction(0,1,precision, 1)
f.TFdirac(0.25)

print(find_nxor([alpha], f, memory, [quality], [1, 100], 0.997, 0.0001, True))


