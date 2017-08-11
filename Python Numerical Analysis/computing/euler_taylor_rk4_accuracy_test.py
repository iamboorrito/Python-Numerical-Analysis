'''
Created on May 4, 2017

@author: Evan Burton

Test of accuracy of methods against specific Taylor method
for the ODE y' = 2*y - exp(t).

Euler's Method is as inaccurate as ever, Taylor's Method does adequately, 
and RK4 is almost magically precise.

Results:

actual: 20.0855369231876677
euler:  8.23692695015
taylor: 19.5352957572
rk4:    20.0831047735


'''
from numerical_ode import euler_method, rk4_method
from numpy import zeros, linspace
import matplotlib.pyplot as plt
from math import exp

# This is Taylor's ODE method for the function 2*y - exp(t)
def taylor(f, a, b, h, y0):
    
    n = int( (b-a)/h ) + 1
    
    w = zeros(n)
    
    w[0] = y0
    t = a
    
    for i in range(1, n):
        
        w_p = w[i-1]
        
        w[i] = w_p + h*f(t, w_p) + h*h*( 2*f(t, w_p)-exp(t) )/2.0
        t += h
    
    return w

# ODE: y' = f(t, y)
def f(t, y):
    return 2*y - exp(t)

# Solve using euler, taylor 2, and rk4
y_euler = euler_method(f, 0, 3, .1, 1)
y_taylor = taylor(f, 0, 3, 0.1, 1)
y_rk4 = rk4_method(f, 0, 3, .1, 1)

# Print solns to console
print('euler:', y_euler[y_euler.shape[0]-1])
print('taylor:', y_taylor[y_taylor.shape[0] - 1])
print('rk4:', y_rk4[y_rk4.shape[0] - 1])

# Set up x/t-axis
x = linspace(0, 3, y_euler.shape[0])

# Plot the solutions
plt.plot(x, y_euler, label='Euler', marker='>')
plt.plot(x, y_taylor, label='Taylor', marker='.')
plt.plot(x, y_rk4, label='RK4')

plt.legend()
plt.show()

