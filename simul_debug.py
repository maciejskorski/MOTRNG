#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *

alpha=0.5
memory = 10
precision = 1000


f = TimeFunction(0,1,precision, 1)
f.TFsquare(alpha)
f.scale = 2

s1 = TimeFunction(0,1,precision, 1)
s1.TFsquare(alpha)
s1.scale = 1

s0 = TimeFunction(0,1,precision, 1)
s0.TFsquare(alpha,True)
s0.scale = 1

g = TimeFunction(0,1,precision, 1)
g.TFgaussian(0, 0.02)

root = TreeNode(f)
listleaves=[]
root.buildtree(memory,s0,s1,g,listleaves)
print listleaves

