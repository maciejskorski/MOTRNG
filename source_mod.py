#!/usr/bin/python
import sys
import math
from math import *

"""
Evaluate in t a function f whose Fourrier series is given by tab
INPUT :
tab : Fourrier series of f
t : t real
OUTPUT :
f(t)
"""
def frequencytotime(tab,t):
    sum = tab[0]
    for i in range(1,len(tab)):
        sum = sum + tab[i]*sin(2*pi*i*t)
    return sum

"""
Compute square signal f(t) in frequency domain
INPUT :
f(t) = 1 if 0<t<alpha
f(t) = 0 if alpha < t < 1
n : number of terms of the Fourrier series
alpha : duty cicle
OUTPUT : f a list of Fourrier coefficients
"""
def signal_carre(n, alpha):
    f=[0.5]
    for i in range(1,n):
        f.append(1/(2*pi*i)*(cos(2*pi*i*alpha)-cos(0))
    return f

"""
Plot a function f given by its Fourrier series
INPUT :
tab : Fourrier series of f
min, max : intervalle
nb_points : number of points
name : name of the curve
"""
def plot(tab, min, max, nb_points,name):
    output = open(name,'w')
    for i in range(nb_points):
        x = min + float(i)*(max-min)/float(nb_points)
        output.write(str(x))
        output.write(" ")
        output.write(str(frequencytotime(tab,x)))
        output.write("\n")
    output.close()

"""
Numerical intégration
INPUT :
tab : a Fourrier series
min, max : the sum interval
nb_points : number of points of the sum
OUTPUT : sum between min and max of f (function associated to tab)
"""
def sum(tab, min, max, nb_points):
    sum = 0
    step = (max-min)/float(nb_points)
    for i in range(nb_points):
        x = min + i*step
        sum = sum + frequencytotime(tab,x)*step
    return sum
 
"""
Effect of time evolution on the probability distribution
INPUT :
tab : a Fourrier series of the probability distribution
variance : volatility^2 of the Wiener process
OUTPUT :
tab1 : the Fourrier series of the new probability distribution
"""
def time_evolution(tab, variance):
    tab1=[]
    for i in range(len(tab)):
        tab1.append(tab[i]*exp(-2*pi**2*variance*i**2))
    return tab1

"""
Sampling effect on the probability distribution
INPUT :
tab : a Fourrier series of the probability distribution
tab1 : a Fourrier series of the sampling function
OUTPUT :
tab2 : a Fourrie serries
"""

A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
"""
def choose(n, k):
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

def xor(p,n):
    sum = 0
    for i in range(1,n,2):
        sum = sum + choose(n,i)*p**i*(1-p)**(n-i)
    return sum

def entropy(p):
    return -p*log(p,2)-(1-p)*log(p,2)

def biais(tab):
    return 2*sum(tab, 0.5, 1,1000)

def find_biais(tab, min, max, nb_points, bias, n_xor):
    x = min
    for i in range(nb_points):
        mtab = time_evolution(tab, x)
        last = biais(mtab)
        last = xor(last,n_xor)
        x = min + float(i)*(max-min)/float(nb_points)
        mtab = time_evolution(tab, x)
        next = biais(mtab)
        next = xor(next,n_xor)

        if (last < bias) and (next > bias):
            return x



