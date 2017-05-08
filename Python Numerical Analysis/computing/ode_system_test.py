'''
Created on May 8, 2017

@author: Evan Burton
'''

from numpy import array
from computing.numerical_ode import rk4_method_m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lorenz Attractor
def f(t, y):
    return array([
        10*(y[1] - y[0]),
        y[0]*(24 - y[2]) - y[1],
        y[0]*y[1] - 8*y[2]/3.0
    ])
 
# Initial value
y_0 = array([
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