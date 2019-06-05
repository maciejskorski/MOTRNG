#!/usr/bin/python
import sys
import math
from math import *

from source_mod_time import *
precision = 1000
quality_factor = 0.229

g = TimeFunction(0,1,precision, 1)
g.TFgaussian(0, quality_factor)

print g.TFsum()
g.TFplot("temp/graph.txt")



