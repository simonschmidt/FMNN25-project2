#!/usr/bin/env python
import scipy
import pylab

class OptimizationProblem(object):
    """ ??? """
    def __init__(self, f, gradient = None):
        """Solves an optimization problem, except that it doesn't

        Arguments:
        f -- the function
        gradient -- the gradient (default to numerical approximation)
        """

        self.f = f
        if gradient:
            self.gradient = gradient
        else:
            def df(x):
                return scipy.derivative(self.f,x)
            self.gradient = df

    def argmax(self,start=None,range=None):
        """Finds the input that gives the maximum value of f"""
        pass

    def argmin(self,start=None,range=None):
        """Finds the input that gives the minimum value of f"""
        pass

    def max(self,start=None,range=None):
        """Finds the maximum of f"""

        return self.f(self.argmax())

    def min(self,start=None,range=None):
        """Finds the minimum of f"""

        return self.f(self.argmin())
