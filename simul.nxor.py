#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
memory = 20
precision = 1000

for i in range(1,18):
    f = TimeFunction(0,1,precision, 1)
    f.TFdirac(0.25)
    print(i)
    print("Dirac")
    print trng_entropy([alpha], f, i, 1, [0.0025],False)

    f = TimeFunction(0,1,precision, 1)
    f.TFconst(1)
    print("Cst")
    print trng_entropy([alpha], f, i, 1, [0.0025],False)


