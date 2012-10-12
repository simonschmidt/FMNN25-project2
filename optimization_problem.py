#!/usr/bin/env python
#coding: utf8
import scipy
import scipy.linalg
import numpy
import pylab
import matplotlib.pyplot as pyplot
import matplotlib.mlab
from chebyquad_problem import *
from pprint import pprint
try:
    from prettytable import PrettyTable
except ImportError:
    pass

class OptimizationProblem(object):
    """ ??? """
    dx = 0.000001
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

    def plot(self,start=None,region=(-5.,5.),title=None):
        if self.shape != 2:
            raise NotImplementedError(u'only capable of plotting functions  ℝ² -> ℝ')
        if start == None:
            start = scipy.zeros((self.shape,))

        pyplot.subplot(211)
        if title:
            pyplot.suptitle(title)
        pyplot.subplots_adjust(hspace=1.)
        pyplot.title('Iteration progress')
        pyplot.xlabel('iteration')
        vals = scipy.zeros((100,2))
        vals[0] = start
        try:
            for i in xrange(1,100):
                vals[i] = self.argmin(start=vals[i-1],maxit=i)
                if numpy.linalg.norm(vals[i]-vals[i-1])<0.000001:
                    break
        except numpy.linalg.LinAlgError as e:
            print e
        vals = vals[:i]
        yvals = [self.f(v) for v in vals]
        pyplot.plot(range(len(yvals)),yvals, label='$f(x_i)$')
        pyplot.legend()

        pyplot.subplot(212)
        # Set up data for contour plot
        #delta=0.05
        #xx=numpy.arange(region[0],region[1],delta)
        #yy=numpy.arange(region[0],region[1],delta)
        xx = scipy.linspace(region[0],region[1],150)
        yy = scipy.linspace(region[0],region[1],150)
        X,Y = numpy.meshgrid(xx,yy)
        z = scipy.zeros((len(yy),len(xx)))

        for i,y in enumerate(yy):
            for j,x in enumerate(xx):
                z[i,j]= self.f([x,y])

        pyplot.subplot(212)
        pyplot.contour(X,Y,z,levels=z[75,range(0,150,3)],colors=pyplot.cm.jet(scipy.linspace(0,1,75)))
        pyplot.title('Contour lines with Search path')
        pyplot.xlabel('$x_1$')
        pyplot.ylabel('$x_2$')
        pyplot.hold(True)
        pyplot.plot(vals[:,0],vals[:,1],'-',label='Search path')
        pyplot.xlabel('$x_1$')
        pyplot.ylabel('$x_2$')
        pyplot.legend(loc=0)

        pyplot.show()

    @classmethod
    def test(cls,start=None,title=None,f=None,gradient=None):
        if not title:
            title = cls.__name__
        # def rosenbrock(x):
        #     return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
        # elnrosen = ExactLineNewton(rosenbrock,2)
        # try:
        #     elnrosen.argmin()
        #     assert False, "Was able to argmin rosenbrock, should have caused problems with positive definiteness"
        # except numpy.linalg.LinAlgError:
        #     pass
        if f==None:
            def f(x):
                return scipy.sqrt((x[0]+2.1)**2 + (x[1]-2.2)**2) + x[0]**2
                #return (x[0]-2.3)**2 + (x[1]+0.2)**2
                #return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
                #return 0.5 * x[0]**2 + 2.5 * x[1]**2
                #return scipy.exp((x[0]+1.)**2+(x[1]-2.3)**2)
        inst = cls(f,2,gradient=gradient)
        inst.plot(start=start,title=title)
        print "Argmin: %s" % inst.argmin(start=start)
        return inst




class Newton(OptimizationProblem):
    # typ klar
    def argmin(self,start=None,tolerance=0.0001,maxit=1000,stepsize=1.0):
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


    def argminAdaptive(self,start=None,tolerance=0.0001,maxit=1000,stepsize=1.0):
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

class BroydenNewton(Newton):
    def argmin(self, start=None, tolerance=0.0001, maxit=1000, stepsize=1.0):
        x_old = start if start != None else scipy.zeros(self.shape)

        inverse = scipy.linalg.inv(self._approx_hess()(x_old))
        
        x_new = x_old - scipy.dot(inverse, x_old)

        for it in xrange(1, maxit):
            g_x = x_old - x_new

            if numpy.linalg.norm(g_x) < tolerance: break

            g_f = self.gradient(x_new)

            lhs_num = g_x - scipy.dot(inverse, g_f)
            lhs_den = scipy.dot(scipy.dot(g_x.T, inverse), g_f)

            lhs = lhs_num / lhs_den
            rhs = scipy.dot(g_x.T, inverse)

            inverse = inverse + scipy.dot(lhs, rhs)

            (x_old, x_new) = (x_new, x_new - stepsize * scipy.dot(inverse, x_new))

        return x_new

class BadBroydenNewton(Newton):
    def argmin(self, start=None, tolerance=0.0001, maxit=1000, stepsize=1.0):
        x_old = start if start != None else scipy.zeros(self.shape)

        inverse = scipy.linalg.inv(self._approx_hess()(x_old))
        
        x_new = x_old - scipy.dot(inverse, x_old)

        for it in xrange(1, maxit):
            g_x = x_old - x_new

            if numpy.linalg.norm(g_x) < tolerance: break

            g_f = self.gradient(x_new)

            lhs_num = g_x - scipy.dot(inverse, g_f)
            lhs_den = scipy.dot(g_f.T, g_f)

            lhs = lhs_num / lhs_den
            rhs = g_f.T

            inverse = inverse + scipy.dot(lhs, rhs)

            (x_old, x_new) = (x_new, x_new - stepsize * scipy.dot(inverse, x_new))

        return x_new

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
        #abest = self.lineNewton.argmin(start=0.1)
        return (abest,x+abest*direction)

    def argmin(self,start=None,tolerance=0.001,maxit=1000):
        if start==None:
            start = scipy.zeros(self.shape)
        xold = start
        xnew = self.linesearch(xold)[1]
        for it in xrange(maxit):
            if numpy.linalg.norm(xold-xnew)<tolerance:
                break
            (xold,xnew) = (xnew,self.linesearch(xnew)[1])
        return xnew

    def plot(self,start=None,region=(-4.,4.),title=None):
        if self.shape != 2:
            raise NotImplementedError(u'only capable of plotting functions  ℝ² -> ℝ')
        if start == None:
            start = scipy.zeros((self.shape,))
        (abest,xnext) = self.linesearch(start)

        # Set up data for contour plot
        delta=0.05
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
        pyplot.title('Contour lines with first search line')
        pyplot.xlabel('$x_1$')
        pyplot.ylabel('$x_2$')

        # Starting point
        pyplot.plot(start[0],start[1],'o',label='$x_0$')
        # Search line
        direction=numpy.dot(self.hessian(start),self.gradient(start))
        pyplot.plot([start[0]-0.5*abest*direction[0],start[0]+1.5*abest*direction[0]],
                    [start[1]-0.5*abest*direction[1],start[1]+1.5*abest*direction[1]],'-',label="search line")
        # Best point
        pyplot.plot(xnext[0],xnext[1],'o',label="$x_0 + a^* * direction$")
        pyplot.legend(loc=0)



        pyplot.subplot(212)
        pyplot.hold(True)
        pyplot.xlabel('a')
        xx=scipy.linspace(0,2*abest)
        pyplot.plot(xx,[self.f(start + ai*direction) for ai in xx],label="Value on search line")
        pyplot.plot(abest,self.f(start + abest*direction),'o')
        pyplot.legend()


        pyplot.figure()
        super(ExactLineNewton,self).plot(start=start,region=region,title=title)
        



class DFP(Newton):
    def argmin(self,start=None,tolerance=0.0001,maxit=1000,stepsize=1.0):
        xold = start if start != None else scipy.zeros(self.shape)

        #B = scipy.identity(self.shape)
        B = self.hessian(xold)
        for it in xrange(maxit):
            if (it != 0 and numpy.linalg.norm(s)<tolerance): break
            ngrad = -1*self.gradient(xold)
            s = numpy.linalg.solve(B,ngrad)
            xnew = xold + s
            y = self.gradient(xnew) + ngrad
            stb = numpy.dot(s,B)
            bs = numpy.dot(B,s)
            B = B + numpy.outer(y,y)/numpy.dot(y,s) - numpy.outer(bs,stb)/numpy.dot(stb,s)
            xold = xnew
        return xnew

class BFGS(Newton):
    def argmin(self,start=None,tolerance=0.0001,maxit=1000):
        xold = start if start != None else scipy.zeros(self.shape)

        B = scipy.identity(self.shape)
        
        for it in xrange(maxit):
            ngrad = -1*self.gradient(xold)

            # TODO: Tolerance break here
            # 

            s = numpy.dot(B,ngrad)

            # Use scipy line search until implemented here
            a=scipy.optimize.linesearch.line_search_wolfe2(\
                self.f,\
                self.gradient,\
                xold,\
                s,\
                -1*ngrad\
                )
            s = a[0] * s
            xnew = xold + s

            # Break when update gives worse value
            # (Line search should have given error)
            if self.f(xnew)>self.f(xold):
                xnew=xold
                break
            # And when getting nan
            if numpy.isnan(xnew).any():
                xnew=xold
                break
            y = self.gradient(xnew) + ngrad

            # Update hessian approximation
            # Using Sherman-Morisson updating
            ytb = numpy.dot(y,B)
            ys = numpy.dot(s,y)
            ss = numpy.dot(s,s)
            by = numpy.dot(B,y)

            B = B + (ys + numpy.dot(ytb,y))*ss/(ys*ys) - (numpy.dot(by,s)+numpy.dot(s,ytb))/ys
            xold = xnew
        return xnew

def test():
    def f(x):
        return x[0]**2+x[1]**2+x[2]**2
    newt = Newton(f,3)
    print newt.hessian([1.,1.,1.])
    print newt.argmin(start=[1.0,2.0,1.5])

def chebquad_test(n=2,start=None,digits=4):
    if start==None:
        start = scipy.rand(n)

    # Store result like [['method1', argmin, f(argmin)],..]
    result=[]

    # Calculate for each class
    for cls in [Newton,BroydenNewton,BadBroydenNewton,ExactLineNewton,DFP,BFGS]:
        try:
            arg = cls(f=chebyquad,shape=n,gradient=gradchebyquad).argmin(start=start)
            fmin=chebyquad(arg)
            arg=map(lambda x: round(x,digits),arg)
            result.append([cls.__name__,arg,fmin])
        except Exception as e:
            result.append([cls.__name__,e,'-'])


    # For scipy BFGS
    fminres=scipy.optimize.fmin_bfgs(f=chebyquad,x0=start,fprime=gradchebyquad,full_output=True,disp=True)
    result.append([
        'fmin_bfgs',
        map(lambda x: round(x,digits),fminres[0]),
        fminres[1] ])

    print "Cheb test n=%d start=%s" % (n,start)
    if 'PrettyTable' in globals():
        pt = PrettyTable(['Method', 'Argmin', 'f'])
        for r in result:
            pt.add_row(r)
        print pt
    else:
        pprint(result)

    return result

