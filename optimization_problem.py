#!/usr/bin/env python
#coding: utf8
import scipy
import pylab

class OptimizationProblem(object):
    """ ??? """
    def __init__(self, f, shape, gradient = None):
        """Solves an optimization problem, except that it doesn't

        Arguments:
        f -- the function
        gradient -- the gradient (default to numerical approximation)
        shape -- Dimension of input argument to function
                 (if f: ℝⁿ->ℝ then shape=n)
        """

        self.f = f
        if gradient:
            self.gradient = gradient
        else:
            def df(x):
                return scipy.derivative(self.f,x)
            self.gradient = df
        self.shape = shape

    def argmax(self,start=None):
        """Finds the input that gives the maximum value of f
            Optional arguments:
                start: Starting point
                range: Search range
        """
        pass

    def argmin(self,start=None):
        """Finds the input that gives the minimum value of f
            Optional arguments:
                start: Starting point
                range: Search range
        """
        pass

    def max(self,start=None):
        """Finds the maximum of f
            Optional arguments:
                start: Starting point
                range: Search range
        """

        return self.f(self.argmax(),start)

    def min(self,start=None):
        """Finds the minimum of f
            Optional arguments:
                start: Starting point
                range: Search range
        """
        return self.f(self.argmin(), start)
