#!/usr/bin/python
import sys
import math
import copy
import numpy as np
from math import *

"""
Class of time functions
"""
class TimeFunction:
    """
    Init function of a class of functions of time.
    INPUT :
        - [min, max] definition interval of the function ;
        - prec : number of points of the function
        - scale : scale factor
        - val : array of values of the function
    """
    def __init__(self, min, max, prec=1000, scale = 1, val=[]):
        self.min = min
        self.max = max
        self.prec = prec
        self.scale = 1
        if len(val) != prec:
            self.val = [0 for i in range(prec)]
        else:
            self.val = val

    """
    Evaluate a time function f in a point x (f is suppose to be periodic of periode max-min)
    INPUT :
        - x
    OUTPUT :
        - min, max in interval such that f(x) \in [min, max]
    """
    def TFeval(self, x, debug=False):

        l=self.max-self.min
        if (x >= self.max) or (x < self.min):
            x=x-math.floor((x-self.min)/l)*l
            if x==self.max:
                x=self.min
        step = l/float(self.prec)
        k= int(math.floor((x-self.min)/step))
        if debug:
            print x,step, x/step, type(x/step),int(x/step), x/step-251.0, x-0.251, step-0.001
        return self.val[k]

    """
    Multiplication by a scalar
    INPUT :
        - f a time function
        - c a real number
    OUTPUT :
        - c*f
    """
    def TFcmult(self, c):
        self.scale = self.scalre *c

    """
    Compute the sum of a time function
    INPUT : 
        - f a time function
    OUTPUT :
    if prov = True
        - dmin, dmax an interval such that sum_[min, max] f \in [dmin, dmax]
    else
        - sum_[min, max] f
    """
    def TFsum(self,prov=False):
        l = (self.max-self.min)/float(self.prec)
        my_sum_min = 0
        my_sum_max = 0
        my_sum =0
        for i in range(len(self.val)):
            if i == (len(self.val)-1):
                my_min, my_max = self.val[i], self.val[i]
            else:
                my_min, my_max = min(self.val[i], self.val[i+1]), max(self.val[i], self.val[i+1])
            my_sum_min = my_sum_min + my_min*l
            my_sum_max = my_sum_max + my_max*l
            my_sum = my_sum + self.val[i]*l
        if prov:
            return my_sum_min*self.scale, my_sum_max*self.scale
        else:
            return my_sum*self.scale

    """
    Compute the entropy of a time function
    INPUT : 
        - f a time function
    OUTPUT :
        - sum_[min, max] f log f dx
    """
    def TFentropy(self, fact=1):
        l = (self.max-self.min)/float(self.prec)
        my_sum =0
        for i in range(len(self.val)):
            prob = self.val[i]*l 
            my_sum = my_sum + prob * log(prob)/log(2)    
        return my_sum*self.scale*fact

    """
    Compute the product of two times functions
    INPUT :
        - self, f 2 times functions
    OUTPUT :
        - self*f such that sum_[min,max] self*f =1
    """
    def TFprod(self, f):
        if (self.prec != f.prec) or (self.min != f.min) or (self.max != f.max):
            print "Error"
            return 0
        else:
            f1 = TimeFunction(self.min, self.max, self.prec, self.scale)
            for i in range(self.prec):
                f1.val[i]=self.val[i]*f.val[i]
            f1.scale = f.scale*self.scale
            return f1

    """
    Plot a time function f given by its Fourrier series
    INPUT :
    tab : Fourrier series of f
    min, max : intervalle
    nb_points : number of points
    name : name of the curve
    """
    def TFplot(self,name, min=None , max=None):
        if min == None:
            min = self.min
        if max == None:
            max = self.max
        output = open(name,'w')
        for i in range(self.prec):
            x = min + float(i)*(max-min)/float(self.prec)
            output.write(str(x))
            output.write(" ")
            output.write(str(self.val[i]))
            output.write("\n")
        output.close()

    """
    Compute the addition of two times functions
    INPUT :
        - self, f 2 times functions
    OUTPUT :
        - self+f such that sum_[min,max] self*f =1
    """
    def TFadd(self, f):
        if (self.prec != f.prec) or (self.min != f.min) or (self.max != f.max):
            print "Error"
            return 0
        else:
            f1 = TimeFunction(self.min, self.max, self.prec, self.scale)
            for i in range(self.prec):
                f1.val[i]=self.val[i]+f.val[i]
            my_sum,_  = TFsum()
            f1.scale = 1/self.TFsum()
            return f1

    """
    Compute the max norm 2 times functions
    INPUT :
        - self, f 2 times functions
    OUTPUT :
        - returns max(self(x)-f(x))
    """
    def TFmaxnorm(self,f):
        if (self.prec != f.prec) or (self.min != f.min) or (self.max != f.max):
            print "Error"
            return 0
        else:
            my_max = 0
            for i in range(self.prec):
                x = abs(f.val[i]*f.scale-self.val[i]*self.scale)
                if x > my_max:
                    my_max = x
            return my_max

    """
    Compute the convolution product of two times functions self and f
    self is supposed to be periodic of period (self.max-self.min)
    This is a slow algorithm
    INPUT :
        - self, f 2 times functions
    OUTPUT :
        - self*f such that sum_[min,max] self*f =1
    """
    def TFconvslow(self, f):
        if (self.prec == f.prec) and ((self.max-self.min) == (f.max-f.min)):
            f1=TimeFunction(self.min, self.max, self.prec, 1)
            sf = (f.max - f.min)/float(f.prec)
            for i in range(f.prec):
                sum = 0
                count = 0
                for j in range(f.prec):
                    k = int(i-j-f.min/sf)
                    sum = sum+f.val[j]*self.val[k]*sf
                f1.val[i]=sum
                f1.scale = f.scale*self.scale
            return f1
        else:
            print "Bad precision of TFcon()"
            return 0

    """
    Compute the convolution product of two times functions self and f
    self is supposed to be periodic of period (self.max-self.min)
    INPUT :
        - self, f 2 times functions
    OUTPUT :
        - self*f such that sum_[min,max] self*f =1
    """
    def TFconv(self, f, epsilon = 0, debug=False):
        if (self.prec == f.prec) and ((self.max-self.min) == (f.max-f.min)):
            f1=TimeFunction(self.min, self.max, self.prec, 1)
            f_fft = np.fft.fft(f.val)
            self_fft = np.fft.fft(self.val)
            conv_fft=[]
            for i in range(len(f_fft)):
                conv_fft.append(f_fft[i]*self_fft[i])
            conv = np.fft.ifft(conv_fft)
            for i in range(len(conv)):
                f1.val[i]=conv[i].real/f1.prec
            f1.scale = f.scale*self.scale
            if debug:
                f2=self.TFconvslow(f)
                if f2.TFmaxnorm(f1) > epsilon:
                    print "Warning TFconv may be false", f2.TFmaxnorm(f1)
                    f2.TFplot("f2.txt")
                    f1.TFplot("f1.txt")
            return f1
        else:
            print "Bad precision of TFcon()"
            return 0

    """
    Returns the Gaussian distribution
    INPUT :
        - mean : mean of the Gaussian distribution
        - sigma : standard deviation
    OUTPUT :
        - f a Gaussian distribution
    """
    def TFgaussian(self, mean, sigma):
        self.scale=1
        k = int((mean - self.min)/(self.max - self.min))
        mean0 = mean - k * (self.max - self.min)
        t = (self.max-self.min)/float(self.prec)
        for i in range(self.prec):
            x=self.min +t*i
            my_range = int(5*(self.max-self.min)/sigma)
            self.val[i]=0
            for j in range(-my_range,my_range):
                x0 = x +j*(self.max-self.min)
                self.val[i]=self.val[i]+1/(sigma*sqrt(2.0*pi))*math.exp(-1/2.0*((x0-mean0)/sigma)**2)

    """
    Returns a constant function on the [sef.min, self.max] interval 
    INPUT :
        - x : value of the constant
    """
    def TFconst(self,x):
        self.val=[x for i in range(self.prec)]


    """
    Returns a square function on the [sef.min, self.max] interval that is

    f(x)= 1 if 0 < x < (self.max-self.min)*alpha+self.min 
    f(x)=0 if  (self.max-self.min)*alpha+self.min  < x < 1
    if inv = False
    or 
    1-f(x) if inv = True
    INPUT :
        - alpha : duty cicle
        - inv : compute rather 1-f(x)
    """
    def TFsquare(self,alpha, inv=False):
        self.val=[0 for i in range(self.prec)]
        if inv:
            for i in range(int(self.prec*alpha)+1, self.prec):
                self.val[i]=1
        else:
            for i in range(int(self.prec*alpha)):
                self.val[i]=1

    """
    Returns a Dirac distribution on the [self.min, self.max] interval that is


    f(t)= self.prec with t=self.min +(self.max-self.min) x for i in [0,1] 
    f(t)=0 otherwise
    INPUT :
        - x : position of the Dirac in [0, 1]
    """
    def TFdirac(self,x):
        if (x >= 0) and (x <= 1):
            self.val=[0 for i in range(self.prec)]
            self.val[int(self.prec*x)]=self.prec
        else:
            print "Error bad parameter in Dirac"

"""
Tree class
"""
class TreeNode:
    """Init Tree class
    INPUT :
    - timefunction : a time function
    - depth : integer depth of the node (number of steps from root)
    - parent : parent node (None for root)
    - childs : list of childs
    - str : an integer name of the node
    """
    def __init__(self, timefunction, depth=0, parent=None, childs=[],label=0):
        self.tf=timefunction
        self.depth = depth
        self.parent = parent
        self.childs = childs
        self.label = label


    """
    Build recurcively a tree to represent the markov chain associated to a TRNG
    INPUT :
        node : a treenode
        
        depth : current depth of the tree
        finaldepth : maximaldepth of the tree
        f : a timefunction
        s0 : timefunction sampling by 0
        s1 : timefunction sampling by 1
        g : evoluation law

    """
    def buildtree(self, finaldepth, s0,s1,g, listleaves, epsilon=0.05, plot=False):
        f=self.tf
        if type(listleaves) != list or len(listleaves) < 2**finaldepth:
            for i in range(2**finaldepth):
                listleaves.append(0)
        depth = self.depth
        if plot:
            name = "temp/test" + str(self.label) +".txt"
            f.TFplot(name)
        if depth == finaldepth:
            prob_min, prob_max = f.TFsum(True)
            if (prob_max-prob_min)>epsilon:
                print(prob_max-prob_min)
                print("Warning the precision is not good enouth")
            moy=(prob_min+prob_max)/2
            if moy > 0.5:
                if prob_max > 1:
                    prob_max = 1
                listleaves[self.label]=prob_max
            else:
                if prob_min > 1:
                    prob_min = 1
                listleaves[self.label]=prob_min
        else:
            mysum = f.TFsum()
            f.scale = f.scale/mysum
            f=f.TFconv(g)
            if plot:
                f.TFplot("temp/graph1.txt")
                g.TFplot("temp/graph.txt")

            f0=f.TFprod(s0)
            f1=f.TFprod(s1)
            self.childs=[ TreeNode(f0, depth+1, self,[], 2*self.label), TreeNode(f1, depth+1,
                self,[],2*self.label+1)]
            self.childs[0].buildtree(finaldepth, s0, s1, g, listleaves)
            self.childs[1].buildtree(finaldepth, s0, s1, g, listleaves)

"""
Information source class
"""
class Info:
    """
    Class Info init
    INPUT :
    - mem : memory of the Markov chain
    - listlongstates : list of probability of memory+1 patterns
    - stable_state : boolean which tells if the stable state has been computed
    """
    def __init__(self, mem, longstates, stable_state = False):
        self.mem = mem
        self.ls=longstates
        self.stable_state = stable_state


    def __str__(self):
        mystr='['
        for i in range(2**(self.mem+1)):
            mystr = mystr + str(self.ls[i]) + ','
        mystr = mystr+']'
        return mystr


    """
    Build a markov chain from a tree
    INPUT :
        self : a markov chain
        listleaves : list of pairs [n,x] where n is mem+1 lenght pattern and x is its
        probability

    OUTPUT : 
    self : with trans computed
    """
    def treetomarkov(self, listleaves, check=True, epsilon=0.01):
        self.ls = []
        for i in range(2**(self.mem+1)):
            if check and i % 2 == 1:
                self.ls.append(float(listleaves[i-1])*2**(-self.mem))
                self.ls.append(2**(-self.mem)*(1-float(listleaves[i-1])))
                if abs(listleaves[i] + listleaves[i-1] -1) > epsilon:
                    print(abs(listleaves[i] + listleaves[i-1] -1))
                    print("Warning the precision is not good enouth")

        return 1
    """
    Compute 
    INPUT :
        self : an markov chain 
        state : state

    OUTPUT : 
    the probability of the state
    """
    def proba_state(self, s):
        return self.ls[s*2]+self.ls[s*2+1]

    """
    Compute 
    INPUT :
        self : an markov chain 
        state : state
        b=0,1 : a transition

    OUTPUT : 
    the probability of transition of the state
    """
    def proba_trans(self, s,b):
        proba_state = self.proba_state(s)
        if proba_state != 0:
            return self.ls[2*s +b]/proba_state
        else:
            return 0

    """
    Compute the advance of a Markov chain
    INPUT :
        self : an markov chain 

    OUTPUT : 
        self : with state updated
    """
    def advance(self):
        if self.mem > 0:
            new_proba = [0]*2**(self.mem+1)
            for state in range(2**(self.mem)):
                prec_state1= int(state/2)+2**(self.mem-1)
                prec_state0 = int(state/2)
                trans = state % 2
                new_proba_state = self.proba_state(prec_state0)*self.proba_trans(prec_state0,trans)+self.proba_state(prec_state1)*self.proba_trans(prec_state1,trans)
                new_proba[2*state]=new_proba_state*self.proba_trans(state,0)
                new_proba[2*state+1]=new_proba_state*(1-self.proba_trans(state,0))
            self.ls = new_proba
        return 1       

    """
    Compute stable state
    INPUT :
        self : an markov chain 

    OUTPUT : 
    self : stable state computed
    """
    def stablestate(self,precision=0.001, debug = False):
        flag = True
        while(flag):
            mem = [ self.ls[j] for j in range(len(self.ls))]
            self.advance()
            flag=False
            for i in range(len(self.ls)):
                if abs(self.ls[i] - mem[i])*2**self.mem > precision:
                    flag = True
        self.stable_state= True
        return 1

    """
    Compute the entropy of a Markov chain
    """
    def entropy(self):
        if not(self.stable_state):
            self.stablestate()
        sum=0
        for state in range(2**self.mem):
            p=self.proba_trans(state,0)
            if (p<=0) or (p>=1):
                ent = 0
            else:
                ent = -p * log(p)/log(2) - (1-p)* log(1-p)/log(2)
            sum=sum + self.proba_state(state)*ent
        return sum

    """
    Compute the xor of two Markov chains
    """
    def markovxor(self, b):
        if not(self.stable_state):
            self.stablestate()
        if not(b.stable_state):
            b.stablestate()

        proba_ls = [0]*(2**(self.mem+1))

        for s in range(len(self.ls)):
            sum = 0
            for s1 in range(len(self.ls)):
                sum = sum + self.ls[s1 ^ s]*b.ls[s1]
            proba_ls[s]=sum
        xor = Info(self.mem, proba_ls)

        return xor

    """
    Compute the n times the xor of the self Markov chain
    """
    def nmarkovxor(self, n):
        if not(self.stable_state):
            self.stablestate()
        double = copy.deepcopy(self)
        ntimes = None
        while n >0:
            if n%2 == 1:
                if ntimes == None:
                    ntimes = double
                else:
                    ntimes = double.markovxor(ntimes)
            n=n/2
            if n > 0:
                double = double.markovxor(double)
        return ntimes


"""
Compute the entropy of a TRNG obtain the n times the xor of the same elementary TRNG
INPUT :
    - alpha : a list the duty cycle of the elementary TRNG
    - f : distribution of probability representing the knowledge of the attacker at the
      begining
    - memory >= 1: the memory of the markov chain that simulate the elementary TRNG
    - nxor : number of branchs of elementary TRNG which are xored
    - qualityfactor : list of quality factor (if len de this list is 1 then we compute the
      xor of nxor elementary TRNG)
OUTPUT : 
    - the entropy of the TRNG
"""
def trng_entropy(alpha, f, memory, nxor, qualityfactor, debug=False):
    precision = f.prec

    if (len(qualityfactor)==1) and (len(alpha)==1):
        s1 = TimeFunction(0,1,precision, 1)
        s1.TFsquare(alpha[0])
        s1.scale = 1

        s0 = TimeFunction(0,1,precision, 1)
        s0.TFsquare(alpha[0],True)
        s0.scale = 1

        g = TimeFunction(0,1,precision, 1)

        g.TFgaussian(0, sqrt(qualityfactor[0]))

        root = TreeNode(f)
        ll=[]
        if debug:
            print("Build tree...")
        root.buildtree(memory+1,s0,s1,g,ll)

        myinfo = Info(memory, [])
        myinfo.treetomarkov(ll)
        if debug:
            print("Computing the xor")
            print(nxor)
        xorn = myinfo.nmarkovxor(nxor)
        if debug:
            print(myinfo)
            print(xorn)

    else:
        if len(qualityfactor) == 1:
            my_len = len(alpha)
            qualityfactor = [qualityfactor[0] for i in range(my_len)]
        if len(alpha) == 1:
            my_len = len(qualityfactor)
            alpha = [alpha[0] for i in range(my_len)]
        if len(qualityfactor) != len(alpha):
            print "Error len(qualityfactor) != len(alpha)"
            return 0
        else:
            my_len = len(alpha)
            for i in range(my_len):
                s1 = TimeFunction(0,1,precision, 1)
                s1.TFsquare(alpha[i])
                s1.scale = 1

                s0 = TimeFunction(0,1,precision, 1)
                s0.TFsquare(alpha[i],True)
                s0.scale = 1

                g = TimeFunction(0,1,precision, 1)
                g.TFgaussian(0, sqrt(qualityfactor[i]))

                root = TreeNode(f)
                ll=[]
                if debug:
                    print("Build tree...")
                root.buildtree(memory+1,s0,s1,g,ll)

                myinfo = Info(memory,[])
                myinfo.treetomarkov(ll)

                if debug:
                    print("Computing the xor")
                    print(i)

                if i == 0:
                    xorn = copy.deepcopy(myinfo)
                    print(myinfo.entropy())
                else:
                    xorn = xorn.markovxor(myinfo)
                if debug:
                    print(xorn.entropy())

    return xorn.entropy(), xorn


"""
Compute a quality factor for a target entropy
INPUT :
    - alpha : a list of duty cycle of the elementary TRNG
    - f : distribution of probability representing the knowledge of the attacker at the
      begining
    - memory : the memory of the markov chain that simulate the elementary TRNG
    - nxor : number of branchs of elementary TRNG which are xored
    - slopes : a list of slopes given by the internal method of CHESS 2014 (variance of the jitter
      after T2 waiting time rescalled to 1)
    if len(slopes)==1 we compute the entropy of nxor xor elementary TRNG
    - span : interval of wainting time to look for
    - target : target entropy
    - epsilon : precision
OUTPUT : the quality factor to obtain the target entropy at precision precision
"""
def find_waiting_time(alpha, f, memory, nxor, slopes, span,target, epsilon=0.0001, debug=False):
    """
    returns the list of qualityfactors
    """
    def spantoquality(span,slopes):
        qualityfactor=[]
        for j in range(len(span)):
            qualityfactor.append([])
            for i in range(len(slopes)):
                qualityfactor[j].append(slopes[i]*span[j])
        return qualityfactor

    if span[1] < span[0]:
        span = [span[1], span[0]]
    qualityfactor=spantoquality(span, slopes)
    ent1,_=trng_entropy(alpha, f, memory, nxor, qualityfactor[0], debug)
    ent2,_=trng_entropy(alpha, f, memory, nxor, qualityfactor[1], debug)
    spane=[ent1, ent2]
    if (target > spane[1]) or (target < spane[0]):
        print "Error, target not in the quality factor interval"
        print spane[0], spane[1], target
        return None
    while (spane[1]-spane[0] > epsilon) and (span[1]-span[0] >= 1):
        if debug:
            print spane[0], spane[1]
        nspan= span[0]+math.floor((span[1]-span[0])/2.0)
        qualityfactor=spantoquality([nspan], slopes)
        newent,_=trng_entropy(alpha, f, memory, nxor, qualityfactor[0], debug)
        if target > newent:
            span=[nspan, span[1]]
            spane=[newent, spane[1]]
        else:
            span=[span[0], nspan]
            spane=[spane[0], newent]
    return span, spane

"""
Compute a quality factor for a target entropy
INPUT :
    - alpha : a list of duty cycle of the elementary TRNG
    - f : distribution of probability representing the knowledge of the attacker at the
      begining
    - memory : the memory of the markov chain that simulate the elementary TRNG
    - nxor : number of branchs of elementary TRNG which are xored
    - slopes : a list of slopes given by the internal method of CHESS 2014 (variance of the jitter
      after T2 waiting time rescalled to 1)
    if len(slopes)==1 we compute the entropy of nxor xor elementary TRNG
    - span : interval of nxor to look for
    - target : target entropy
    - epsilon : precision
OUTPUT : the number of xor to obtain the target entropy at precision precision
"""
def find_nxor(alpha, f, memory, quality, span,target, epsilon=0.0001, debug=False):
    if span[1] < span[0]:
        span = [span[1], span[0]]

    spane = []
    for i in range(2):
        ent,_=trng_entropy(alpha, f, memory, span[i], quality, debug)
        spane.append(ent)

    if (target > spane[1]) or (target < spane[0]):
        print "Error, target not in the quality factor interval"
        print spane[0], spane[1], target
        return None
    while (spane[1]-spane[0]> epsilon) and (span[1]-span[0] > 1):
        if debug:
            print spane[0], spane[1]
        nspan= int(span[0]+math.floor((span[1]-span[0])/2.0))
        newent,_=trng_entropy(alpha, f, memory, nspan, quality, debug)
        if target > newent:
            span=[nspan, span[1]]
            spane=[newent, spane[1]]
        else:
            span=[span[0], nspan]
            spane=[spane[0], newent]
    return span, spane


