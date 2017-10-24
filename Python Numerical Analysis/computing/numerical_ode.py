'''
Created on May 2, 2017
First Order Numerical ODE Solvers

Implemented:

Array-valued functions:
    Euler's Method:          euler_method(...)
    Midpoint Method:         midpoint_method(...)
    Modified Euler's Method: modified_euler_method(...)
    RK4:                     rk4_method(...)
                             rk4_method_m(...)
                             rk4_m(...)
    
Real-valued functions:
    Euler's Method:          euler(...)
    Midpoint Method:         midpoint(...)
    Modified Euler's Method: modified_euler(...)

Each function has an argument list as follows:
    
    method(f, a, b, h, y0)
    
    f is a python function f(t, y)
        where y = y(t) unless it ends with _m, then it is a 
        vector-valued function which accepts and returns a
        numpy.ndarray
        
    a is the initial value for t
    b is the final value for t
    h is the step size
    y0 is the initial value y(a) = y0
    
Each function that ends with FUNC_NAME_method returns a numpy array
    w of size int( (b-a)/h + 1 ) where w[0] = y(a) and w[k] = y(a+k*h). 
    
Each function that is just euler(...), midpoint(...), rk4_m(...) only 
    returns the final value y(b) if solving on the interval [a, b].
    

###############################################################################
                    Example use with pyplot: y' = f(t, y)
###############################################################################

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
    
###############################################################################
                    Example use with pyplot: 3D System
###############################################################################

# import numpy to use its arrays
import numpy as np
# import rk4_method_m for solving matrix-vector DEs with RK4
from numerical_ode import rk4_method_m
# matplotlib for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

""" 
    Functions should take 2 arguments (t, y) where t is a scalar and y can be a
    a vector. For this example, we have the Lorenz system,
    
    | y0 |'   | 10 y1 - 10 y0      |
    | y1 |  = | 24 y0 - y0*y2 - y1 |
    | y2 |    | y0*y1 - 8/3 y2     |
    
"""
def f(t, y):
    return np.array([
        10*(y[1] - y[0]),
        y[0]*(24 - y[2]) - y[1],
        y[0]*y[1] - 8*y[2]/3.0
    ])
 
# Initial value
y_0 = np.array([
        1,
        1,
        1
    ])

# t in [0, 70]
a = 0
b = 70
 
h = 0.01
 
result = rk4_method_m(f, a, b, h, y_0)

# Get x, y, z solutions 
y1 = result[:, 0]
y2 = result[:, 1]
y3 = result[:, 2]
 
fig = plt.figure()
fig.add_subplot(111, projection='3d')
 
plt.plot(y1, y2, y3, linestyle='--')
plt.show()


@author: Evan Burton
'''

from numpy import zeros, array, ndarray

'''
Euler's method for solving y' = f(t, y), this
returns an array of values for y(t) on [a, b].
'''
def euler_method(f, a, b, h, y0):
    
    # Add one to account for initial value
    n = int( (b-a)/(h)) + 1
    
    w = zeros(n)
    w[0] = y0
    
    t = a
    
    for i in range(1, n):
        
        m1 = h*f(t, w[i-1])
        
        w[i] = w[i-1] + m1
            
        #print w[i]
        t += h
    
    return w

'''
Euler's method which only returns y(b).
'''
def euler(f, a, b, h, y0):
    
    # Add one to account for initial value
    n = int( (b-a)/(h)) + 1
    
    wi = y0
    
    t = a
    
    for i in range(1, n):
        
        m1 = h*f(t, wi)
        
        wi += m1
            
        t += h
    
    return wi

'''
Modified Euler's method for solving y' = f(t, y), this
returns an array of values for y(t) on [a, b].
'''
def modified_euler_method(f, a, b, h, y0):
    
    n = int( (b-a)/(h)) + 1
    
    w = zeros(n)
    w[0] = y0
    t = a
    
    for i in range(1, n):
        
        wi = w[i-1]
        m1 = h*f(t, wi)
        m2 = h*f(t+h, wi+m1)

        w[i] = wi+(m1+m2)/2.0
            
        t += h
    
    return w
    
'''
Modified Euler's method which only returns the
final value y(b).
'''
def modified_euler(f, a, b, h, y0):
    
    n = int( (b-a)/(h)) + 1
    
    wi = y0
    t = a
    
    for i in range(1, n):
        
        m1 = h*f(t, wi)
        m2 = h*f(t+h, wi+m1)
        
        wi += (m1+m2)/2.0
            
        t += h
    
    return wi
    
'''
Midpoint method for solving y' = f(t, y). This
returns an array of values for y(t) on [a, b].
'''
def midpoint_method(f, a, b, h, y0):
    
    n = int( (b-a)/(h)) + 1
    
    w = zeros(n)
    w[0] = y0
    
    t = a
    
    for i in range(1, n):

        wi = w[i-1]
        
        m1 = wi+h*f(t,wi)/2.0
        m2 = h*f(t + h/2.0, m1)
        
        w[i] = wi + m2
    
        t += h
    
    return w

'''
Midpoint method which only returns the final value y(b).
'''
def midpoint(f, a, b, h, y0):
    
    n = int( (b-a)/(h)) + 1
    
    wi = y0
    
    t = a
    
    for i in range(1, n):
        
        m1 = wi+h*f(t,wi)/2.0
        m2 = h*f(t + h/2.0, m1)
        
        wi = wi + m2
    
        t += h
    
    return wi
'''
RK4 method for solving y' = f(t, y). This
returns an array of values for y(t) on [a, b].
'''
def rk4_method(f, a, b, h, y0):
    
    # Get number of points
    n = int( (b-a)/(h)) + 1
    
    # Initialize solution vector
    w = zeros(n)
    
    # Set initial values
    w[0] = y0
    t = a
    
    for i in range(1, n):
        
        # Only read w[i-1] once per iteration
        wi = w[i-1]
        
        # RK4 Scheme
        k1 = h*f(t, wi)
        k2 = h*f(t + h/2.0, wi+k1/2.0)
        k3 = h*f(t + h/2.0, wi+k2/2.0)
        k4 = h*f(t+h, wi+k3)
        
        w[i] = wi+(k1+2*(k2+k3)+k4)/6.0
            
        t += h
        
    return w

'''
RK4 for solving the matrix differential equation
    Y' = F(t, Y) with output as an N x M numpy
    array, where N = # of points and M = # of rows
    of Y'.
    
    For a 2D system, result = [Y1, Y2]
    
    Result = [Y1, Y2, ..., YM]
'''
def rk4_method_m(f, a, b, h, y0):
    
    # Get number of points
    n = int( (b-a)/(h)) + 1
    
    if type(y0) is ndarray:
        y_initial = y0
    else:
        y_initial = array([y0])
    
    # Initialize solution vector
    wi = y_initial
    result = zeros((n, y_initial.shape[0]))
    
    for j in range(wi.shape[0]):
        result[0, j] = wi[j]
    t = a
    
    for i in range(1, n):
        
        # RK4 Scheme
        k1 = h*f(t, wi)
        k2 = h*f(t + h/2.0, wi+k1/2.0)
        k3 = h*f(t + h/2.0, wi+k2/2.0)
        k4 = h*f(t+h, wi+k3)
        
        wi = wi + (k1+2*(k2+k3)+k4)/6.0

        for j in range(wi.shape[0]):
            result[i, j] = wi[j]

        t += h
        
    return result

'''
Alternate to rk4_method_m which only returns the 
final value Y(b).
'''
def rk4_m(f, a, b, h, y0):
    
    # Get number of points
    n = int( (b-a)/(h)) + 1
    
    if type(y0) is ndarray:
        w = y0
    else:
        w = array([y0])
    
    t = a
    
    for i in range(1, n):
        
        # RK4 Scheme
        k1 = h*f(t, w)
        k2 = h*f(t + h/2.0, w+k1/2.0)
        k3 = h*f(t + h/2.0, w+k2/2.0)
        k4 = h*f(t+h, w+k3)
        
        w = w + (k1+2*(k2+k3)+k4)/6.0

        t += h
        
    if w.shape[0] == 1:
        return w[0]
    
    return w
