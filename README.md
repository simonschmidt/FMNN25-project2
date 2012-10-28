FMNN25-project2
===============

http://www.maths.lth.se/na/courses/FMNN25/media/material/project02_.pdf

# Todo
* ~~Optimization problem class~~
* ~~Classical Newton~~
* ~~Newton + Exact line search~~
* ~~Inexact line search~~
* ~~Task 7~~
* ~~Provide newton's method with inexact line search option~~
* ~~Good Broyden~~
* ~~Bad Broyden~~
* ~~DFP~~
* ~~BFGS~~
* ~~chebquad~~  (chebquad_test)
* ~~compare fmin_bfgs~~ (chebquad_test)
* ~~study goodness of BFGS hessian approximation~~ (hessian_test)


Test on Rosenbrock Function
# Exact line search using newton method
--
On the function
```python
def f(x):
    return scipy.sqrt((x[0]-1.1)**2 + (x[1]-3.2)**2) + x[0]**2
```
Doing a line search starting at point (0,0) by using newton method to find
best position on the line segment in the direction of the gradient the newton
method becomes unstable and will not converge
see Non-converging-newton-line-function.png  and Non-converging-newton.png
Was fixed by using argminAdaptive that checks if new point is worse than old one
and in that case decrease stepsize

# How-to!
 1. Chose your favorite solver among the following:
 * Newton (Newton)
 * Broyden-Newton (BroydenNewton)
 * Bad Broyden-Newton (BadBroydenNewton)
 * Exact Line search (ExactLineNewton)
 * Inexact line search (InexactLineMethod)
 * DFP (DFP)
 * BFGS (BFGS)

 2. Define your function and, if you want, also the gradient and the hessian. If those are not given a numerical approximation will be used. 

 3. Initiate an instance of your chosen solver with the name in parentheses after the name and then in parantheses give your function, the dimension of the function (if f: R^n -> R then the dimension is n), the gradient if any and lastly the hessian if any. 

 4. To find the minimum argument the runt argmin on your instance. The different argmin methods takes different inputs but they all take the starting point as the first one. 
