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
                return numpy.array([(self.f(x+df.h[i]*self.dx/2.) - self.f(x-df.h[i]*self.dx/2.))/self.dx for i in xrange(shape)])
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
    def argmin(self,start=None,tolerance=0.0001,maxit=1000,stepsize=0.1):
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


    def argminAdaptive(self,start=None,tolerance=0.0001,maxit=1000,stepsize=0.1):
        """ Newton iteration adapting stepsize when overstepping local minima
        """
        if start == None:
            start = scipy.zeros(self.shape)
        xold = start
        xnew = xold - numpy.linalg.solve(self.hessian(xold),self.gradient(xold))
        for it in xrange(maxit):
            if numpy.linalg.norm(xold - xnew)<tolerance:
                break
            chol = scipy.linalg.cho_factor(self.hessian(xold))
            delta = scipy.linalg.cho_solve(chol,self.gradient(xold))
            if self.f(xold-stepsize*delta)>self.f(xold):
                stepsize=stepsize*0.8
            (xold, xnew) = (xnew,xold - stepsize*delta)
        return xnew

class ExactLineNewton(OptimizationProblem):
    lineNewton=None

    def linesearch(self,x):
        """Do one linesearch step, uses newton method to find correct region
            Arguments: 
                x: Starting point
            Returns:
                (abest, xnext)
        """
        # Search direction
        direction = numpy.dot(self.hessian(x),self.gradient(x))

        # f restricted to the line
        fline = lambda a: self.f(x+a*direction)

        # Reuse newton class, 
        if self.lineNewton:
            self.lineNewton.f = fline
        else:
            self.lineNewton = Newton(fline, 1)

        abest = self.lineNewton.argminAdaptive(start=0.1)
        return (abest,x+abest*direction)

    def argmin(self,start=None,tolerance=0.001,maxit=1000):
        if not start:
            start = scipy.zeros(self.shape)
        xold = start
        xnew = self.linesearch(xold)[1]
        for it in xrange(maxit):
            if numpy.linalg.norm(xold-xnew)<tolerance:
                break
            (xold,xnew) = (xnew,self.linesearch(xnew)[1])
        return xnew

    def plot(self,startx=None,region=(-4.,4.)):
        if self.shape != 2:
            raise NotImplementedError(u'only capable of plotting functions  ℝ² -> ℝ')
        if not startx:
            startx = scipy.zeros((self.shape,))
        (abest,xnext) = self.linesearch(startx)

        # Set up data for contour plot
        delta=0.025
        xx=numpy.arange(region[0],region[1],delta)
        yy=numpy.arange(region[0],region[1],delta)
        X,Y = numpy.meshgrid(xx,yy)
        z = scipy.zeros((len(yy),len(xx)))

        for i,y in enumerate(yy):
            for j,x in enumerate(xx):
                z[i,j]= self.f([x,y])
        pyplot.subplot(211)
        pyplot.hold(True)
        pyplot.contour(X,Y,z)

        # Starting point
        pyplot.plot(startx[0],startx[1],'o',label='$x_0$')
        # Search line
        direction=numpy.dot(self.hessian(startx),self.gradient(startx))
        pyplot.plot([startx[0]-0.5*abest*direction[0],startx[0]+1.5*abest*direction[0]],
                    [startx[1]-0.5*abest*direction[1],startx[1]+1.5*abest*direction[1]],'-',label="search line")
        # Best point
        pyplot.plot(xnext[0],xnext[1],'o',label="$x_0 + a^* * direction$")
        pyplot.legend(loc=0)


        pyplot.subplot(212)
        pyplot.hold(True)
        xx=scipy.linspace(0,2*abest)
        pyplot.plot(xx,[self.f(startx + ai*direction) for ai in xx],label="Value on search line")
        pyplot.plot(abest,self.f(startx + abest*direction),'o')
        pyplot.legend()
        pyplot.show()

    @classmethod
    def test(cls):
        # def rosenbrock(x):
        #     return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
        # elnrosen = ExactLineNewton(rosenbrock,2)
        # try:
        #     elnrosen.argmin()
        #     assert False, "Was able to argmin rosenbrock, should have caused problems with positive definiteness"
        # except numpy.linalg.LinAlgError:
        #     pass

        def f(x):
            return scipy.sqrt((x[0]-1.1)**2 + (x[1]-3.2)**2) + x[0]**2
            #return (x[0]-2.3)**2 + (x[1]+0.2)**2
        eln = ExactLineNewton(f,2)
        eln.plot()
        print "Argmin: %s" % eln.argmin()


def test():
    def f(x):
        return x[0]**2+x[1]**2+x[2]**2
    newt = Newton(f,3)
    print newt.hessian([1.,1.,1.])
    print newt.argmin(start=[1.0,2.0,1.5])
