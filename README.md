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
    
** There is no rk4(...) yet, but rk4_m(...) should suffice **

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
