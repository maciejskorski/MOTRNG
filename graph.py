#!/usr/bin/python
import sys
import math
from math import *

from source_mod import *

min = 0

max = 0.001
nb_points = 100
n_xor = 16
n_xor1 = 32
n_xor2 = 64
bias = 0.499

tab = signal_carre(30)
step = float((max-min))/nb_points

output = open("graph.txt", "w");
output1 = open("graph1.txt","w");
output2 = open("graph2.txt","w");

output3 = open("graph_entropy1.txt","w");
output4 = open("graph_entropy2.txt","w");
output5 = open("graph_entropy3.txt","w");


def log2(x):
    return log(x)/log(2)


def entropy(p):
    return -p*log2(p)-(1-p)*log2(1-p)


variance = min
for i in range(nb_points):
    variance = variance + step
    mtab = time_evolution(tab,variance)
    p = biais(mtab)
    p1 = xor(p,n_xor)
    p2 = xor(p,n_xor1)
    p3 = xor(p,n_xor2)

    e1=entropy(p1)
    e2=entropy(p2)
    e3=entropy(p3)

    output.write(str(variance))
    output.write(" ")
    output.write(str(p))
    output.write("\n")

    output1.write(str(variance))
    output1.write(" ")
    output1.write(str(p2))
    output1.write("\n")

    output2.write(str(variance))
    output2.write(" ")
    output2.write(str(p3))
    output2.write("\n")

    output3.write(str(variance))
    output3.write(" ")
    output3.write(str(e1))
    output3.write("\n")

    output4.write(str(variance))
    output4.write(" ")
    output4.write(str(e2))
    output4.write("\n")

    output5.write(str(variance))
    output5.write(" ")
    output5.write(str(e3))
    output5.write("\n")


output.close()
output1.close()
output2.close()

#print find_biais(tab, min, max, 100000, bias, 16)
#print find_biais(tab, min, max, 100000, bias, 32)
#print find_biais(tab, min, max, 100000, bias, 64)
