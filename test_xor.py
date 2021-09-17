#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha = 0.5
memory = 15
precision = 1000
sigma = 0.08

s1 = TimeFunction(0,1,precision, 1)
s1.TFsquare(alpha)
s1.scale = 1

s0 = TimeFunction(0,1,precision, 1)
s0.TFsquare(alpha,True)
s0.scale = 1

g = TimeFunction(0,1,precision, 1)
g.TFgaussian(0, sigma)

f = TimeFunction(0,1,precision, 1)
f.TFdirac(0.25)
g = TimeFunction(0,1,precision, 1)
g.TFgaussian(0, sqrt(memory *sigma**2))
f=f.TFconv(g)
f = f.TFprod(s1)
print("Accumulation interne")
print(f.TFsum())

sum = 0
for i in range(2**memory):
    binstr= bin(i)[2:]
    binstr = '0'*(memory-len(binstr))+binstr

    xor = 0
    for k in range(len(binstr)):
        xor = xor ^ int(binstr[k])

    if xor == 0:
        f = TimeFunction(0,1,precision, 1)
        f.TFdirac(0.25)
        f.TFconst(1)
        for l in range(len(binstr)):
            f=f.TFconv(g)
            if binstr[l]== '0':
                f=f.TFprod(s0)
            else:
                f=f.TFprod(s1)
#        f.TFplot("temp/graph"+binstr+"txt")
        sum = sum + f.TFsum()
print("Accumulation avec xor")
print(sum)


    

