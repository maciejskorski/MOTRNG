#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.4917
avance = 500
memory = 3

f = TimeFunction(0,1,1000, 1)
f.TFsquare(alpha)
f.scale = 2

s1 = TimeFunction(0,1,1000, 1)
s1.TFsquare(alpha)
s1.scale = 1

s0 = TimeFunction(0,1,1000, 1)
s0.TFsquare(alpha,True)
s0.scale = 1

g = TimeFunction(-10,10,1000, 1)
g.TFgaussian(0, 0.02)

root = TreeNode(f)
listleaves=[]
root.buildtree(memory,s0,s1,g,listleaves)
print listleaves

print "entropy 0"
print trng_entropy(alpha, memory, 29, 0.02**2)
info = Info()
node = info.treetomarkov(listleaves)
print info.listnodesname
print info.listnodes
matrix = info.markovtomatrix()
print info.matrix
n=info.stablestate()
print n
print "entropy"
print info.entropy()
xor = info.markovxor(info)
print info.listnodes
print xor.listnodes
print "xor entropy"
print xor.entropy()
n1 = xor.stablestate()
print n1
xorn = info.nmarkovxor(29)
print xorn.listnodes
print "xorn"
print xorn.entropy()
