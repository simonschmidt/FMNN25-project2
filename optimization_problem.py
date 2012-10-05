#!/usr/bin/env python
#coding: utf8
import scipy
import scipy.linalg
import numpy
import pylab

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
                return [(f(x+df.h[i]*self.dx/2.) - f(x-df.h[i]*self.dx/2.))/self.dx for i in xrange(shape)]
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

    def linesearch(self):
        pass

def test():
    def f(x):
        return x[0]**2+x[1]**2+x[2]**2
    newt = Newton(f,3)
    print newt.hessian([1.,1.,1.])
    print newt.argmin(start=[1.0,2.0,1.5])
