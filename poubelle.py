    """
    Compute the convolution product of two times functions self and f
    self is supposed to be periodic of period (self.max-self.min)
    INPUT :
        - self, f 2 times functions
    OUTPUT :
        - self*f such that sum_[min,max] self*f =1
    """
    def TFconv(self, f):
        f1=TimeFunction(self.min, self.max, self.prec, self.scale)
        step = (f.max-f.min)/float(f.prec)
        for i in range(f1.prec):
            x=f1.min + (f1.max- f1.min)/float(f1.prec)*i
            sum = 0
            for j in range(f.prec):
                t=f.min + (f.max- f.min)/float(f.prec)*j
                sum = sum+f.val[j]*self.TFeval(x-t)*step
            f1.val[i]=sum
            f1.scale = f.scale*self.scale
        return f1

 
