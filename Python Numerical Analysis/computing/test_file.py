'''
Created on May 4, 2017

@author: Evan Burton
'''
from computing.numerical_ode import euler_method, rk4_method
from numpy import zeros, linspace
import matplotlib.pyplot as plt
from math import exp

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

