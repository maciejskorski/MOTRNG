#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
avance = 500

f = TimeFunction(0,1,1000, 1)
f.TFsquare(alpha)
f.scale = 2

s1 = TimeFunction(0,1,1000, 1)
s1.TFsquare(alpha)
s1.scale = 1

s0 = TimeFunction(0,1,1000, 1)
s0.TFsquare(alpha,True)
s0.scale = 1

g = TimeFunction(-1,1,1000, 1)
g.TFgaussian(0, 0.01)

for i in range(avance):
    f = f.TFconv(g)
    print f.TFsum()
    f1 = f.TFprod(s1)
    sum1= f1.TFsum()
    f0 = f.TFprod(s0)
    f1.TFplot("testf1"+str(i)+".txt")
    f0.TFplot("testf0"+str(i)+".txt")
    sum0= f0.TFsum()
    print "probabilite 1"
    print sum1
    print "probabilite 0"
    print sum0
    if sum1 > sum0:
        f=f1
        f.scale = f.scale /sum1
    else:
        f=f0
        f.scale = f.scale /sum0
    print f.TFsum()
    f.TFplot("test"+str(i)+".txt")


