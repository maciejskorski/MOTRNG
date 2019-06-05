#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

precision = 1000

f = TimeFunction(0,1,precision, 1)
f.TFdirac(0.25)

g = TimeFunction(0,1,precision, 1)
g.TFgaussian(0, 0.1)


#my_fft=np.fft.fft(f.val)
#print my_fft

#my_ifft=np.fft.ifft(my_fft)
#print my_ifft
#print len(my_ifft)
#print len(f.val)
#
#epsilon = 10**-10
#for i in range(len(f.val)):
#    if abs(f.val[i]-my_ifft[i].real) > epsilon:
#        print i, f.val[i], my_ifft[i].real, abs(f.val[i]-my_ifft[i].real) 
fg_conv = f.TFconv(g, 0.0001, True)
