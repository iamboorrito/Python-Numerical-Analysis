'''
Created on May 2, 2017
First Order Numerical ODE Solvers

Implemented:

Array-valued functions:
    Euler's Method:          euler_method(...)
    Midpoint Method:         midpoint_method(...)
    Modified Euler's Method: modified_euler_method(...)
    RK4:                     rk4_method(...)
    
Real-valued functions:
    Euler's Method:          euler(...)
    Midpoint Method:         midpoint(...)
    Modified Euler's Method: modified_euler(...)
    RK4:                     rk4(...)


Each function has an argument list as follows:
    
    method(f, a, b, h, y0)
    
    f is a real-valued function f(t, y)
        where y = y(t)
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

Example use with pyplot at bottom.

@author: Evan Burton
'''

from numpy import zeros, linspace, array, ndarray

def euler_method(f, a, b, h, y0):
    
    # Add one to account for initial value
    n = int( (b-a)/(h)) + 1
    
    w = zeros(n)
    w[0] = y0
    
    t = a
    
    for i in range(1, n):
        
        m1 = f(t, w[i-1])
        
        if m1 > 10**15:
            w[i] = 10**15
        else:
            w[i] = w[i-1]+h*m1
            
        #print w[i]
        t += h
    
    return w

def modified_euler_method(f, a, b, h, y0):
    
    n = int( (b-a)/(h)) + 1
    
    w = zeros(n)
    w[0] = y0
    t = a
    
    for i in range(1, n):
        
        wi = w[i-1]
        m1 = h*f(t, wi)
        m2 = h*f(t+h, wi+m1)
        
        if m1 > 10**15:
            w[i] = 10**15
        else:
            w[i] = wi+(m1+m2)/2.0
            
        t += h
    
    return w
    
def midpoint_method(f, a, b, h, y0):
    
    n = int( (b-a)/(h)) + 1
    
    w = zeros(n)
    w[0] = y0
    
    t = a
    
    for i in range(1, n):

        wi = w[i-1]
        
        m1 = wi+h*f(t,wi)/2.0
        m2 = h*f(t + h/2.0, m1)
        
        if m1 > 10**15:
            w[i] = 10**15
        else:
            w[i] = wi + m2
    
        t += h
    
    return w

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
        
        # Don't allow overflow
        if k1 > 10**15:
            w[i] = 10**15
        else:
            w[i] = wi+(k1+2*(k2+k3)+k4)/6.0
    
            #print w[i]
            
        t += h
    return w

def taylor(f, a, b, h, y0):
    
    n = int( (b-a)/(h)) + 1
    
    w = zeros(n)
    w[0] = y0
    
    t = a
    
    for i in range(1, n):
        
        m1 = h*f(t, w[i-1])
        m2 = h*(4*w[i-1]-3*exp(t))
        
        if m1 > 10**15:
            w[i] = 10**15
        else:
            w[i] = w[i-1]+m1 + h*m2/2.0
            
        t += h
    
    return w


def euler(f, a, b, h, y0):
    
    # Add one to account for initial value
    n = int( (b-a)/(h)) + 1
    
    wi = y0
    
    t = a
    
    for i in range(1, n):
        
        m1 = f(t, wi)
        
        if m1 > 10**15:
            wi = 10**15
        else:
            wi += h*m1
            
        t += h
    
    return wi

def modified_euler(f, a, b, h, y0):
    
    n = int( (b-a)/(h)) + 1
    
    wi = y0
    t = a
    
    for i in range(1, n):
        
        m1 = h*f(t, wi)
        m2 = h*f(t+h, wi+m1)
        
        if m1 > 10**15:
            wi = 10**15
        else:
            wi += (m1+m2)/2.0
            
        t += h
    
    return wi
    
def midpoint(f, a, b, h, y0):
    
    n = int( (b-a)/(h)) + 1
    
    wi = y0
    
    t = a
    
    for i in range(1, n):
        
        m1 = wi+h*f(t,wi)/2.0
        m2 = h*f(t + h/2.0, m1)
        
        if m1 > 10**15:
            wi = 10**15
        else:
            wi = wi + m2
    
        t += h
    
    return wi

def rk4_method_m(f, a, b, h, y0):
    
    # Get number of points
    n = int( (b-a)/(h)) + 1
    
    if type(y0) is ndarray:
        y_initial = y0
    else:
        y_initial = array([y0])
    
    # Initialize solution vector
    wi = y_initial
    result = zeros((n+1, y_initial.shape[0]))
    
    for j in range(wi.shape[0]):
        result[0, j] = wi[j]
    t = a
    
    for i in range(1, n+1):
        
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

def rk4(f, a, b, h, y0):
    
    # Get number of points
    n = int( (b-a)/(h)) + 1
    
    if type(y0) is ndarray:
        y_initial = y0
    else:
        y_initial = array([y0])
    
    # Initialize solution vector
    wi = y_initial
    result = zeros((n+1, y_initial.shape[0]))
    
    for j in range(wi.shape[0]):
        result[0, j] = wi[j]
    t = a
    
    for i in range(1, n+1):
        
        # RK4 Scheme
        k1 = h*f(t, wi)
        k2 = h*f(t + h/2.0, wi+k1/2.0)
        k3 = h*f(t + h/2.0, wi+k2/2.0)
        k4 = h*f(t+h, wi+k3)
        
        wi = wi + (k1+2*(k2+k3)+k4)/6.0

        t += h
        
    return wi

###########################################
################## Test ###################
import matplotlib.pyplot as plt
from math import exp

# ODE: y' = f(t, y)
def f(t, y):
    return array([
        y[1],
        y[0]*(1-y[3])+t,
        y[3],
        y[2]**2-y[1]*y[3]+t*(y[0]**2)      
                  ])
    #return exp(-t)

# Initial value y0
y0 = array([1, 2, 3, 4])

# Interval [a, b]
a = 0.0
b = 1

# Step size h
h = 0.01

# Numerically solve IVP

result = rk4_method_m(f, a, b, h, y0)
print result
y1 = result[:, 0]
y2 = result[:, 2]
x = linspace(a, b, y1.shape[0])

# label1 = 'Euler'
# y_1 = euler_method(f, a, b, h, y0)
# label2 = 'Mod. Euler'
# y_2 = modified_euler_method(f, a, b, h, y0)
# label3 = 'RK4'
# y_3 = rk4_method(f, a, b, h, y0)
# 
# x = linspace(a, b, y_3.shape[0])
# 
# # Print final values for y(3)
# print label1, y_1[y_1.shape[0]-1]
# print label2, y_2[y_2.shape[0]-1]
# print label3, y_3[y_3.shape[0]-1]
# 
# # Plot and show solution
# plt.plot(x, y_1, label=label1, linestyle = '-')
# plt.plot(x, y_2, label=label2, linestyle = '-')
# plt.plot(x, y_3, label=label3, linestyle='-')

plt.plot(x, y1)
plt.plot(x, y2)

plt.show()
################ End Test #################
###########################################