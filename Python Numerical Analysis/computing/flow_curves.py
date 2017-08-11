'''
Created on May 8, 2017

@author: Evan Burton
'''

from numpy import array, linspace, sqrt, exp, sin
from computing.numerical_ode import rk4_method, midpoint_method
import matplotlib.pyplot as plt

def f(t, y):
    return sin(y)- t
 
# Initial value
y_0 = 1

# t in [0, 70]
a = 0
b = 1
 
h = .5

mid = rk4_method(f, a, b, h, y_0)
       
print(mid)
 
