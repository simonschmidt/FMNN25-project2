#!/usr/bin/env python
#coding: utf8
import scipy
import scipy.linalg
import numpy
import pylab
import matplotlib.pyplot as pyplot
import matplotlib.mlab

class OptimizationProblem(object):
    """ ??? """
    dx = 0.00001
    def __init__(self, f, shape, gradient = None, hessian = None):
        """Solves an optimization problem, except that it doesn't

        Arguments:
        f -- the function
        shape -- Dimension of input argument to function
                 (if f: ℝⁿ->ℝ then shape=n)
        gradient -- the gradient (default to numerical approximation)
        hessian  -- function returning hessian matrix (default to numerical approximation)
        """

        self.f = f
        self.shape=shape
        if gradient:
            self.gradient = gradient
        else:
            def df(x):
                return numpy.array([(f(x+df.h[i]*self.dx/2.) - f(x-df.h[i]*self.dx/2.))/self.dx for i in xrange(shape)])
            df.h = scipy.identity(self.shape)
            self.gradient = df
        if hessian:
            self.hessian = hessian
        else:
            self.hessian = self._approx_hess()


    def _approx_hess(self):
        """ Gives a function that approximates hessian matrix"""
        def fh(x):
            res = scipy.zeros((self.shape,self.shape))
            for row in xrange(self.shape):
                for col in xrange(row,self.shape):
                    res[row,col] = self._second_deriv(row,col,x)
                    if row != col:
                        res[col,row] = res[row,col]
            # TODO pos-definite check
            # worth it to do it here? cho_factor will test for
            # pos-def implicitly
            return res
        return fh

    def _second_deriv(self,i,j,x):
        """ Numerically calculated ∂²f/∂xᵢ∂x_j at point x
        """
        dxi = scipy.zeros(self.shape)
        dxj = scipy.zeros(self.shape)
        dxi[i] = self.dx
        dxj[j] = self.dx
        return (self.f(x+dxi+dxj) - self.f(x+dxi-dxj) - self.f(x-dxi+dxj) + self.f(x-dxi-dxj))/(4.*self.dx*self.dx)


    def argmax(self,start=None):
        """Finds the input that gives the maximum value of f
            Optional arguments:
                start: Starting point
        """
        raise NotImplementedError

    def argmin(self,start=None):
        """Finds the input that gives the minimum value of f
            Optional arguments:
                start: Starting point
        """
        raise NotImplementedError

    def max(self,start=None):
        """Finds the maximum of f
            Optional arguments:
                start: Starting point
        """

        return self.f(self.argmax(start))

    def min(self,start=None):
        """Finds the minimum of f
            Optional arguments:
                start: Starting point
        """
        return self.f(self.argmin(start))




class Newton(OptimizationProblem):
    # typ klar
    def argmin(self,start=None,tolerance=0.001,maxit=1000,stepsize=0.5):
        if start == None:
            start = scipy.zeros(self.shape)
        xold = start
        xnew = xold - numpy.linalg.solve(self.hessian(xold),self.gradient(xold))
        for it in xrange(maxit):
            if numpy.linalg.norm(xold - xnew)<tolerance:
                break
            chol = scipy.linalg.cho_factor(self.hessian(xold))
            (xold, xnew) = (xnew,xold - stepsize*scipy.linalg.cho_solve(chol,self.gradient(xold)))
        return xnew

class ExactLineNewton(OptimizationProblem):

    def linesearch(self,x):
        direction = -self.gradient(x)
        # f restricted to the line
        fline = lambda a: self.f(x+a*direction)
        abest = Newton(fline,1).argmin(start=0.5)
        return (abest,x+abest*direction)

    @classmethod
    def test(cls):
        def f(x):
            return x[0]**2 +3*x[1]**2
        startx=[2.1,1.3]
        eln = cls(f,2)
        (abest,xnext) = eln.linesearch(startx)
        delta=0.025
        xx=numpy.arange(-3.,3.,delta)
        yy=numpy.arange(-3.,3.,delta)
        X,Y = numpy.meshgrid(xx,yy)
        z = scipy.zeros((len(yy),len(xx)))

        for i,y in enumerate(yy):
            for j,x in enumerate(xx):
                z[i,j]= f([x,y])
        pyplot.subplot(211)
        pyplot.hold(True)
        pyplot.contour(X,Y,z)
        pyplot.hold(True)
        # Starting point
        pyplot.plot(startx[0],startx[1],'o',label='$x_0$')
        # Search line
        pyplot.plot([startx[0],startx[0]-0.5*eln.gradient(startx)[0]],
                    [startx[1],startx[1]-0.5*eln.gradient(startx)[1]],'-',label="search line")
        # Best point
        pyplot.plot(xnext[0],xnext[1],'o',label="$f(x_0 + a^* * direction)$")
        pyplot.legend(loc=0)


        pyplot.subplot(212)
        pyplot.hold(True)
        direction = -1*eln.gradient(startx)
        pyplot.plot(scipy.linspace(0,2*abest),[f(startx + ai*direction) for ai in scipy.linspace(0,2*abest)],label="Value on search line")
        pyplot.plot(abest,f(startx + abest*direction),'o')
        pyplot.legend()
        pyplot.show()
        print (abest,xnext)


def test():
    def f(x):
        return x[0]**2+x[1]**2+x[2]**2
    newt = Newton(f,3)
    print newt.hessian([1.,1.,1.])
    print newt.argmin(start=[1.0,2.0,1.5])
