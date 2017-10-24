# Python-Numerical-Analysis

* This is a work in progress. So far Numerical ODEs is the most complete because that is what I found most interesting and relevant towards my coursework at the time *

A simple collection of files (3 so far) which perform standard numerical algorithms such as 

  - Iterative solvers for fixed points and zeros
    * Bisection, Secant, Fixed Point, Aitken's $\Delta^2$, and Steffenson's Methods.
  - Numerical Integration
    * Trapezoid, Simpson, Romberg integration
  - Numerical ODE Solvers
    * Euler, Modified Euler, Midpoint, RK4 and variants detailed below.
    
# For ODEs, the functions are as follows:
    
Implemented:
 - Array-valued functions:
    * Euler's Method:          euler_method(...)
    * Midpoint Method:         midpoint_method(...)
    * Modified Euler's Method: modified_euler_method(...)
    * RK4:                     rk4_method(...)
                             rk4_method_m(...)
                             rk4_m(...)
    
- Real-valued functions:
    * Euler's Method:          euler(...)
    * Midpoint Method:         midpoint(...)
    * Modified Euler's Method: modified_euler(...)
    
** There is no rk4(...), but rk4_m(...) does the same thing **

Each function has an argument list as follows:
    
    method(f, a, b, h, y0)
    
    f is a real-valued function f(t, y)
        where y = y(t) unless it ends with _m, then it is a 
        vector-valued function which accepts and returns a
        numpy.ndarray
        
    a is the initial value for t
    b is the final value for t
    h is the step size
    y0 is the initial value y(a) = y0
    
Each function that ends with FUNC_NAME_method returns a numpy array
    w of size int( (b-a)/h + 1 ) where w[0] = y(a) and w[k] = y(a+kh). 
    
Therefore,
    
    index = int( (t-a)/h )
    
    y(t) = w[index]
    
Each function that is just euler(...), midpoint(...), rk4(...) only 
    returns the final value y(b) if solving on the interval [a, b]
    
# Example 1: y' = f(t, y)

```python
from numerical_ode import euler_method, rk4_method
import numpy as np
import matplotlib.pyplot as plt

# ODE: y' = f(t, y)
def f(t, y):
    return 2*y - np.exp(t)

# Solve using euler and rk4
y_euler = euler_method(f, 0, 3, .1, 1)
y_rk4 = rk4_method(f, 0, 3, .1, 1)

# Print y(b) to console
print('euler:', y_euler[y_euler.shape[0]-1])
print('rk4:', y_rk4[y_rk4.shape[0] - 1])

# Set up y/t-axis
x = np.linspace(0, 3, y_euler.shape[0])

# Plot the solutions
plt.plot(x, y_euler, label='Euler', marker='>')
plt.plot(x, y_rk4, label='RK4')

plt.legend()
plt.show()
```
