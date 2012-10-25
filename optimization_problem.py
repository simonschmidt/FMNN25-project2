#!/usr/bin/env python
#coding: utf8
import scipy
import scipy.linalg
import numpy
import pylab
import matplotlib.pyplot as pyplot
from pprint import pprint
from chebyquad_problem import *
try:
    from prettytable import PrettyTable
except ImportError:
    pass

class OptimizationProblem(object):
    """ Base class for optimization problems, do not use directly """
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

        if f==None:
            def f(x):
                #return scipy.sqrt((x[0]+2.1)**2 + (x[1]-2.2)**2) + x[0]**2
                #return (x[0]-2.3)**2 + (x[1]+0.2)**2
                return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
                #return 0.5 * x[0]**2 + 2.5 * x[1]**2
                #return scipy.exp((x[0]+1.)**2+(x[1]-2.3)**2)
        inst = cls(f,2,gradient=gradient)
        inst.plot(start=start,title=title)
        print "Argmin: %s" % inst.argmin(start=start)
        return inst




class Newton(OptimizationProblem):

    def argmin(self,start=None,tolerance=0.0001,maxit=100,stepsize=1.0,exact=False):
        if exact: 
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
        else:
            return InexactLineMethod(self.f,self.shape,self.gradient,self.hessian,).argmin(start,tolerance,maxit)


    def argminAdaptive(self,start=None,tolerance=0.0001,maxit=100,stepsize=1.0):
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

            if numpy.isnan(g_f).any(): break

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

            if numpy.isnan(g_f).any(): break

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
                    [start[1]-0.5*abest*direction[1],start[1]+1.5*abest*direction[1]],
                    '-',label="search line")
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
        
class InexactLineMethod(OptimizationProblem):

    def linesearch(self,x, maxit=10,t1=9,fbar=0,sigma=0.1,rho=0.01):
        """
        Does one step in the inexact line method to find a small value of f on the line
            Arguments:
                x = starting point
                maxit = number of iterations
            returns:
                (abest, xnext)
        """
        a = numpy.zeros(maxit+1)
        b = numpy.zeros(maxit+1)
        direction = numpy.dot(self.hessian(x),self.gradient(x))
        fline = lambda a: self.f(x+a*direction)
        fgrad = lambda a: numpy.dot(self.gradient(x+a*direction),direction)

        mu = (-fline(0))/(fgrad(0))
        a[1] = mu/2        

        it=0
        for i in xrange(1, maxit):
            it+=1
            fai=fline(a[i])
            if fai<=fbar:
                break
            if fai>fline(0)+a[i]*rho*fgrad(0) or fai>=fline(a[i-1]):
                a[i]=a[i-1]
                b[i]=a[i]
            fPrimAi=fgrad(a[i])
            if abs(fPrimAi) <= -sigma*fgrad(0):
                break
            if fPrimAi>=0:
                a[i]=a[i]
                b[i]=a[i-1]
            if mu<=2*a[i]-a[i-1]:
                a[i+1]=mu
            else:
                # looks at values of f in 10 places in the interval and takes the one that is the smallest
                seq = linspace(min(mu, a[i]+t1*(a[i]-a[i-1])),2*a[i]-a[i-1],10)
                fseq = array([fline(seq[i]) for i in xrange(len(seq))])
                a[i+1] = seq[fseq.argmin()]
        return (a[it], x+a[it]*direction)

    def argmin(self, start=None,tolerance=1e-3,maxit=1000,t1=9,fbar=0,sigma=0.1,rho=0.01):
        if start==None:
            start = scipy.zeros(self.shape)
        xold = start
        xnew = self.linesearch(xold)[1]
        for it in xrange(maxit):
            if numpy.linalg.norm(xold-xnew)<tolerance:
                break
            (xold,xnew) = (xnew,self.linesearch(xnew)[1])
        return xnew

class DFP(Newton):
    def argmin(self,start=None,tolerance=0.0001,maxit=100,stepsize=1.0):
        xold = start if start is not None else scipy.zeros(self.shape)

        # Initial hessian inverse guess
        B = scipy.identity(self.shape)

        grad=(tolerance+1)*scipy.ones(self.shape)
        for it in xrange(maxit):
            if (it != 0 and numpy.linalg.norm(grad)<tolerance): break

            grad = self.gradient(xold)

            # Search direction
            s = numpy.dot(B,-1*grad)

            # Use scipy line search until implemented here
            a=scipy.optimize.linesearch.line_search_wolfe2(
                self.f,
                self.gradient,
                xold,
                s,
                grad
                )
            s = a[0] * s

            xnew = xold + s
            if numpy.isnan(self.f(xnew)): break

            y = self.gradient(xnew) -grad
            ytb = numpy.dot(y,B)
            by = numpy.dot(B,y)
            B = B + numpy.outer(s,s)/numpy.dot(y,s) - numpy.outer(by,ytb)/numpy.dot(ytb,y)
            xold = xnew
        return xnew

class BFGS(Newton):
    def argmin(self,start=None,tolerance=0.0001,maxit=100,call=None):
        """
            Find a minimum

            start: starting point, default 0
            tolerance: break when ||gradient|| is less than
            maxit: iteration limit
            call: function to call at end of each iteration, is passed locals() as argument

        """
        xold = start if start is not None else scipy.zeros(self.shape)

        B = scipy.identity(self.shape)
        
        for it in xrange(maxit):
            grad = self.gradient(xold)

            if (it != 0 and numpy.linalg.norm(grad)<tolerance): break

            s = numpy.dot(B,-1*grad)

            # Use scipy line search until implemented here
            a=scipy.optimize.linesearch.line_search_wolfe2(
                self.f,
                self.gradient,
                xold,
                s,
                grad
                )
            s = a[0] * s
            xnew = xold + s


            # Break on nan
            if numpy.isnan(xnew).any():
                xnew=xold
                break

            y = self.gradient(xnew) - grad

            # Update inverse hessian approximation
            # Using Sherman-Morisson updating
            ytb = numpy.dot(y,B)
            ys = numpy.dot(s,y)
            ss = numpy.outer(s,s)
            by = numpy.dot(B,y)
            bys = numpy.outer(by,s)

            B = B + (1 + numpy.dot(ytb,y)/ys)*ss/ys - (bys+ numpy.transpose(bys))/ys
            xold = xnew
            if call:
                call(locals())
        return xnew

    def hessian_goodness(self,norm=None,plot=True,invhess=None,title=None):
        """ Compare relative error of inverse hessian approximation to actual inverse

            norm: 'fro', inf, -inf, 2, -2
            plot: Plot relive error
            invhess: Exact inverse hessian, by default uses inv(self.hessian(x))
            title: plot title

            returns list of relative errors
        """
        if norm is None:
            norm='fro'

        # Function that gets called each iteration in BFGS process
        # Compares approximation and exact hessian
        def c(loc):
            approx=loc['B']

            if invhess:
                exact = invhess(loc['xnew'])
            else:
                exact = scipy.linalg.inv(self.hessian(loc['xnew']))

            # Calculate relative error
            c.res.append(numpy.linalg.norm(approx-exact,norm)/numpy.linalg.norm(exact,norm))
        c.res=[]

        self.argmin(call=c)

        if plot:
            pyplot.semilogy(c.res,label='$\\frac{||H^* - H||_{%s}}{||H||_{%s}}$' % (norm,norm))
            pyplot.legend(loc=0)
            pyplot.xlabel('iteration')
            pyplot.ylabel('error')
            if title==None:
                pyplot.title('relative error for BFGS inverse hessian approximation')
            else:
                pyplot.title(title)
            pyplot.show()
        return c.res


def chebquad_test(n=2,start=None,digits=4):
    """
        Run the available optimizers on chebyquad function
        start: starting point, default: random uniform in [0,1]**shape
        digits: digits in output
    """

    if start is None:
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

def hessian_test(norm=None):
    """
        Show goodness of BFGS inverse hessian on rosenbrock function
    """
    def f(x):
        return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    def invhess(x):
        (x,y) = x
        return numpy.array([
            [1/(2 + 400*x**2 - 400*y), x/(1 + 200*x**2 - 200*y)],
            [x/(1+200*x**2-200*y),1/200+(2*x**2)/(1+200*x**2-200*y) ]
            ])


    return BFGS(f=f, shape=2).hessian_goodness(invhess=invhess,title='BFGS inverse hessian approximation for rosenbrock function',norm=norm)
