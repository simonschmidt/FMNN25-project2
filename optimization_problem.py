import scipy
import pylab

class OptimizationProblem(object):
    """ ??? """
    def __init__(self, f, gradient = None):
        """Solves an optimization problem, except that it doesn't

        Arguments:
        f -- the function
        gradient -- the gradient (default to ???)
        """

        self.f = f
        if gradient:
            self.gradient = gradient
        else:
            def df(x):
                return scipy.derivative(x)
            self.gradient = df

    def argmax():
        """Finds the input that gives the maximum value of f"""

    def argmin():
        """Finds the input that gives the minimum value of f"""

    def max():
        """Finds the maximum of f"""

        return self.f(self.argmax())

    def min():
        """Finds the minimum of f"""

        return self.f(self.argmin())
