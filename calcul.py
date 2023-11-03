#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.51
avance = 500
memory = 4

f = TimeFunction(0,1,1000, 1)
f.TFsquare(alpha)
f.scale = 2

s1 = TimeFunction(0,1,1000, 1)
s1.TFsquare(alpha)
s1.scale = 1

s0 = TimeFunction(0,1,1000, 1)
s0.TFsquare(alpha,True)
s0.scale = 1

for i in range(1,100):
    #pente = 5.27*10**-7
    pente = 2.16*10**-6

    variance = pente*i*10
    print("KD = ", i*10)
    print(variance, sqrt(variance))
    g = TimeFunction(-1,1,1000, 1)
    g.TFgaussian(0, 0.02)

    root = TreeNode(f)
    listleaves=[]
    root.buildtree(memory,s0,s1,g,listleaves)

    info = Info()
    node = info.treetomarkov(listleaves)
    matrix = info.markovtomatrix()
    n=info.stablestate()
    xorn = info.nmarkovxor(29)
    print(xorn.entropy())
