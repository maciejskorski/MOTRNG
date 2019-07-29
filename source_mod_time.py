#!/usr/bin/python
import sys
import math
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
        - min, max an interval such that sum_[min, max] f \in [min, max]
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
    - depth : depth of the node (number of steps from root)
    - parent : parent node (None for root)
    - childs : list of childs
    - str : name of the node
    """
    def __init__(self, timefunction, depth=0, parent=None, childs=[],mstr=''):
        self.tf=timefunction
        self.depth = depth
        self.parent = parent
        self.childs = childs
        self.str = mstr

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
    def buildtree(self, finaldepth, s0,s1,g, listleaves):
        f=self.tf
        depth = self.depth
        name = "temp/test" + self.str +".txt"
        f.TFplot(name)
        if depth == finaldepth:
            prob_min, prob_max = f.TFsum(True)
            moy=(prob_min+prob_max)/2
            if moy > 0.5:
                if prob_max > 1:
                    prob_max = 1
                listleaves.append([self.str, prob_max])
            else:
                if prob_min > 1:
                    prob_min = 1
                listleaves.append([self.str, prob_min])
        else:
            mysum = f.TFsum()
            f.scale = f.scale/mysum
            f=f.TFconv(g)
            f.TFplot("temp/graph1.txt")
            g.TFplot("temp/graph.txt")

            f0=f.TFprod(s0)
            f1=f.TFprod(s1)
            self.childs=[ TreeNode(f0, depth+1, self,[], self.str+'0'), TreeNode(f1, depth+1,
                self,[],self.str+'1')]
            self.childs[0].buildtree(finaldepth, s0, s1, g, listleaves)
            self.childs[1].buildtree(finaldepth, s0, s1, g, listleaves)

"""
Markov chain class
"""
class MarkovNode:
    """
    INPUT : 
    - prob : probability of the Node
    - childs : list of childs [Node, probability of transition]
    - name : name of the node
    - count : counter (for algorithmic purpose)
    - newprob : is used to update the probability of the Node
    """
    def __init__(self, prob=None, childs=[], name='', count=0, newprob=0):
        self.prob = prob
        self.childs = childs
        self.name = name
        self.count = count
        self.newprob = newprob

    def __repr__(self):
        repr = "["+self.name+","
        if self.prob == None:
            repr = repr + "None"
        else:
            repr = repr + str(self.prob)
        if len(self.childs) != 2:
            repr = repr + "]"
        else:
            repr = repr + ",["+self.childs[0][0].name + ","+str(self.childs[0][1])+"],["+self.childs[1][0].name+","+str(self.childs[1][1])+"]]"
        return repr 

    """
    Compute a list of nodes
    """
    def __listnodes(self, listnodesname, listnodes,count):
        if node.count < count:
            listnodesname.append(node.name)
            listnodes.append(node)
            node.count = count
            for i in range(2):
                listnodes(node.childs[i][0], listnodesnames, listnodes, count)

    """
    Compute a list of nodes
    """
    def listnodes(self):
        listnodesnames=[]
        listnodes=[]
        count =self.count +1
        self.__listnodes(listnodesname, listnodes, count)
        return listnodesnames, listnodes



"""
Information source class
"""
class Info:
    """
    Class Info init
    INPUT :
    - listnode : list of nodes
    - listnodesnames : list of nodes names corresponding to listnode
    - matrix : probability transition matrix
    - node : root node
    """
    def __init__(self, listnodes=[], listnodesname=[], matrix=None, node=None):
        self.listnodes = listnodes
        self.listnodesname=listnodesname
        self.matrix=matrix
        self.node=node

    """
    Build a markov chain from a tree
    INPUT :
        self : an markov chain 

    OUTPUT : 
    self : with listnodes and listnodesname computed
    """
    def treetomarkov(self, listleaves):
        epsilon=0.1
        for i in range(len(listleaves)): #create all the nodes
            nameleave = listleaves[i][0]
            newname = nameleave[:len(nameleave)-1]
            if not(newname in self.listnodesname):
                self.listnodesname.append(newname)
                node = MarkovNode(None, [], newname,0)
                self.listnodes.append(node)

        for i in range(len(self.listnodes)): #create transitions
            name = self.listnodes[i].name
            for j in range(2):
                leavename = name+str(j)
                for k in range(len(listleaves)):
                    if listleaves[k][0] == leavename:
                        nextnode = self.listnodes[self.listnodesname.index(leavename[1:])]
                        if j==0:
                            self.listnodes[i].childs.append([nextnode,listleaves[k][1]])
                        else:
                            prob = self.listnodes[i].childs[0][1]
                            if abs(listleaves[k][1]+ prob-1) > epsilon:
                                print "Warning probability between childs of state"
                                print abs(listleaves[k][1]+ prob-1), listleaves[k][1]
                            self.listnodes[i].childs.append([nextnode,1-prob])

        for i in range(len(self.listnodes)):
            if self.listnodes[i].name == '0'*(len(listleaves[0][0])-1):
                self.node= self.listnodes[i]

    """
    Compute the matrix associated to a Markov chain
    INPUT :
        self : an markov chain 

    OUTPUT : 
    self : matrix computed
    """
    def markovtomatrix(self):
        matrix=[]
        for i in range(len(self.listnodes)):
            matrix.append([0]*len(self.listnodes))
            mynode=self.listnodes[i]
            str0=mynode.childs[0][0].name
            str1=mynode.childs[1][0].name
            matrix[i][self.listnodesname.index(str0)]=mynode.childs[0][1]
            matrix[i][self.listnodesname.index(str1)]=mynode.childs[1][1]
        self.matrix=matrix

    """
    Compute the advance of a Markov chain
    INPUT :
        self : an markov chain 

    OUTPUT : 
        self : with state updated
    """
    def advance(self):
        for i in self.listnodes:
            i.newprob = 0
        for i in self.listnodes:
            for j in i.childs:
                j[0].newprob = j[0].newprob + i.prob*j[1]
        for i in self.listnodes:
            i.prob = i.newprob
        return 1       

    """
    Compute stable state
    INPUT :
        self : an markov chain 

    OUTPUT : 
    self : stable state computed
    """
    def stablestate(self,precision=0.001, debug = False):
        if len(self.listnodes)==1:
            self.listnodes[0].prob =1
        else:
            self.listnodes[0].prob=1
            for i in range(1,len(self.listnodes)):
                self.listnodes[i].prob=0
            flag = True
            while(flag):
                mem = [ self.listnodes[j].prob for j in range(len(self.listnodes))]
                self.advance()
                flag=False
                for i in range(len(self.listnodes)):
                    if abs(self.listnodes[i].prob - mem[i]) > precision:
                        flag = True

            if debug:
                mem = [ self.listnodes[i].prob for i in range(len(self.listnodes))]
                self.stablestateslow()
                for i in range(len(self.listnodes)):
                    if abs(self.listnodes[i].prob - mem[i]) > 2*precision:
                        print "Error"

            return 1

    """
    Compute stable state
    INPUT :
        self : an markov chain 

    OUTPUT : 
    self : stable state computed
    """
    def stablestateslow(self,precision=0.001):
        if len(self.listnodes)==1:
            self.listnodes[0].prob =1
        else:
            if self.matrix == None:
                self.markovtomatrix()
            m= np.array(self.matrix)
            n1=np.array([float(0)]*len(self.matrix))
            n1[0]=1
            n=np.array([float(0)]*len(self.matrix))

            while max([ abs(n[i] - n1[i]) for i in range(len(self.matrix))]) > precision:
                n=n1
                n1=n.dot(m)

            for i in range (len(self.listnodes)):
                self.listnodes[i].prob = n[self.listnodesname.index(self.listnodes[i].name)]
            return 1

    """
    Compute the entropy of a Markov chain
    """
    def entropy(self):
        sum=0
        nan_flag=False
        for i in range(len(self.listnodes)):
            p=self.listnodes[i].childs[0][1]
            if (p<=0) or (p>=1):
                ent = 0
            else:
                ent = -p * log(p)/log(2) - (1-p)* log(1-p)/log(2)
            sum=sum + self.listnodes[i].prob*ent
        return sum

    """
    Compute the xor of two Markov chains
    """
    def markovxor(self, b):
        precision = 0.1
        """
        xor two str names
        """
        def strxor(str1, str2):
            str3=''
            for i in range(len(str1)):
                str3=str3+ str((int(str1[i]) + int(str2[i]))%2)
            return str3


        listpairs=[] #list of pairs of nodes of Markov chains
        for i in range(len(self.listnodes)):
            for j in range(len(b.listnodes)):
                listpairs.append([self.listnodes[i], b.listnodes[j]])
        
        listnodexor= [] #list of pairs of nodes which projects to the same node of the xor
        #Markov chain
        xor = Info([], [], None, None) # the result
        for i in range(len(listpairs)): #compute the names of states of the xor Markov chain
            pair = listpairs[i]
            namexor=strxor(pair[0].name, pair[1].name)
            if not(namexor in xor.listnodesname):
                xor.listnodesname.append(namexor)
                listnodexor.append([ pair])
            else:
                ind = xor.listnodesname.index(namexor)
                listnodexor[ind].append(pair)
        xor.listnodes=[] #list of nodes of the xor Markov chain
        for i in range(len(xor.listnodesname)): #compute the nodes of the xor Markov chain
            node = MarkovNode(None,[], xor.listnodesname[i], 0)
            sum =0.0
            for j in range(len(listnodexor[i])):
                sum = sum + listnodexor[i][j][0].prob * listnodexor[i][j][1].prob
            node.prob=sum
            xor.listnodes.append(node)
        for i in range(len(xor.listnodesname)): #compute the probabilities of the childs
            name = xor.listnodes[i].name
            newnode = []
            for j in range(2):
                if len(name)>1:
                    newname = name[1:len(name)]+str(j)
                else:
                    newname = ''
                newnode.append(xor.listnodes[xor.listnodesname.index(newname)])
            sum=[0.0,0.0]
            sum1 =0
            for k in range(len(listnodexor[i])):
                nodek = listnodexor[i][k]
                prob = listnodexor[i][k][0].prob*listnodexor[i][k][1].prob
                sum1= sum1 + prob
                childprob=[]
                childprob.append(listnodexor[i][k][0].childs[0][1]*listnodexor[i][k][1].childs[0][1]+listnodexor[i][k][0].childs[1][1] *listnodexor[i][k][1].childs[1][1])
                    
                childprob.append(listnodexor[i][k][0].childs[0][1] *
                        listnodexor[i][k][1].childs[1][1] +
                        listnodexor[i][k][0].childs[1][1] *
                        listnodexor[i][k][1].childs[0][1])
                sum = [sum[0] + prob*childprob[0], sum[1] + prob*childprob[1]]

            sum = [sum[0]/xor.listnodes[i].prob, sum[1]/xor.listnodes[i].prob]
            if abs(sum[0]+sum[1]-1) > precision:
                print "Error ", abs(sum[0]+sum[1]-1), " greater than", precision
            else:
                xor.listnodes[i].childs.append([newnode[0], sum[0]])
                xor.listnodes[i].childs.append([newnode[1], 1-sum[0]])
        xor.stablestate()
        return xor

    """
    Compute the n times the xor of the self Markov chain
    """
    def nmarkovxor(self, n):
        double = self
        ntimes = None
        while n >0 :
            if n%2 == 1:
                if ntimes == None:
                    ntimes = double
                else:
                    ntimes = double.markovxor(ntimes)
            double = double.markovxor(double)
            n=n/2
        return ntimes


"""
Compute the entropy of a TRNG obtain the n times the xor of the same elementary TRNG
INPUT :
    - alpha : the duty cycle of the elementary TRNG
    - f : distribution of probability representing the knowledge of the attacker at the
      begining
    - memory : the memory of the markov chain that simulate the elementary TRNG
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
        listleaves=[]
        root.buildtree(memory,s0,s1,g,listleaves)

        info = Info([], [], None, None)
        node = info.treetomarkov(listleaves)

        if debug:
            n=info.stablestate(precision = 0.001, debug = True)
            print info.listnodes
        else:
            n=info.stablestate()


        xorn = info.nmarkovxor(nxor)

        if debug:
            print xorn.listnodes
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
                listleaves=[]
                root.buildtree(memory,s0,s1,g,listleaves)

                info = Info([], [], None, None)
                node = info.treetomarkov(listleaves)
                

                if debug:
                    n=info.stablestate(precision=0.001, debug = True)
                    print info.listnodes
                else:
                    n=info.stablestate()


                if i == 0:
                    xorn = info
                else:
                    xorn = xorn.markovxor(info)
                if debug:
                    print i,xorn.listnodes
        return xorn.entropy()


"""
Compute a quality factor for a target entropy
INPUT :
    - alpha : the duty cycle of the elementary TRNG
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
    - precision : number of points of evaluation of the functions
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
    spane=[trng_entropy(alpha, f, memory, nxor, qualityfactor[0], debug), trng_entropy(alpha, f, memory,
            nxor, qualityfactor[1], debug)]
    if (target > spane[1]) or (target < spane[0]):
        print "Error, target not in the quality factor interval"
        print spane[0], spane[1], target
        return None
    while (spane[1]-spane[0] > epsilon) and (span[1]-span[0] >= 1):
        if debug:
            print spane[0], spane[1]
        nspan= span[0]+math.floor((span[1]-span[0])/2.0)
        qualityfactor=spantoquality([nspan], slopes)
        newent=trng_entropy(alpha, f, memory, nxor, qualityfactor[0], debug)
        if target > newent:
            span=[nspan, span[1]]
            spane=[newent, spane[1]]
        else:
            span=[span[0], nspan]
            spane=[spane[0], newent]
    return span, spane
