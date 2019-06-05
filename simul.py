#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.4917
avance = 500
memory = 3




ratio_pente = [1, 1.27, 1.49, 1.86, 2.29, 2.80, 3.67, 4.83, 6.15, 7.89, 9.39, 11.26, 13.59, 12.27, 15.01,
1.04, 1.28, 1.52, 1.93, 2.39, 2.90, 3.65, 5.33, 6.65, 7.63, 9.84, 11.66, 13.87, 15.92]
facteur = 5.27*10**-7
ratio_pente = [facteur * i for i in ratio_pente]

print find_waiting_time(alpha, memory, 29, ratio_pente, [150, 200], 0.997, 0.001, 1000, True)
