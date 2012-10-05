FMNN25-project2
===============

http://www.maths.lth.se/na/courses/FMNN25/media/material/project02_.pdf


# Exact line search using newton method
On the function
```python
def f(x):
    return scipy.sqrt((x[0]-1.1)**2 + (x[1]-3.2)**2) + x[0]**2
```
Doing a line search starting at point (0,0) by using newton method to find
best position on the line segment in the direction of the gradient the newton
method becomes unstable and will not converge
see Non-converging-newton-line-function.png  and Non-converging-newton.png

