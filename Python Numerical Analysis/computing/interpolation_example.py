'''
Created on Aug 11, 2017

@author: evan
'''
from math import exp
from interpolation import *
from numerical_integration import composite_simpson

def f(x):
    return exp(x)*x - x**2
 
xs = [1, 2, 3]
ys = [f(1), f(2), f(3)]
p = interp_poly(xs, ys)

ch = cheb_poly(f, (0,3))

def cheb(x):
    return ch[x]
def pol(x):
    return p[x]

actual = composite_simpson(f, 0, 6, 1000)
equi_spaced = composite_simpson(pol, 0, 6, 1000)
cheb_nodes = composite_simpson(cheb, 0, 6, 1000)

print('actual: %f\nequi-spaced: %f\nchebyshev: %f\n' % (actual, equi_spaced, cheb_nodes))