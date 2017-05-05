'''
Created on May 2, 2017

@author: Evan Burton
'''

'''
Python 3 recommended fix for cmp(a, b)
'''
def cmp(a, b):
    return (a > b) - (a < b) 

'''
Using the bisection method:

Solves the equation f(x) = 0 on [a, b] to given
tolerance and max iterations.

Returns a tuple(a,b) where b-a < tolerance which
encloses the zero of f(x).
'''
def bisection_method(f, a, b, tol, maxIterations):
    
    i = 0
    fa = f(a)
    fb = f(b)
    
    m = (a+b)/2.0
    
    while abs(fa-fb) > tol and i <= maxIterations:
        if cmp(fa, 0)*cmp(f(m), 0) < 0:
            b = m
        else:
            a = m
            
        m = (a+b)/2.0
        fa = f(a)
        fb = f(b)
       
        i += 1

    if(i == maxIterations):
        print('Max Iterations Exceeded (bisection_method)')
    
    
    return (a,b)

'''
Using fixed-point iteration:

Solves the equation f(x) = x with initial guess p0 
to given tolerance and max iterations.

Returns the fixed point.
'''
def fixed_point_method(f, p0, tol, maxIterations):
    
    p_old = f(p0)
    p_new = f(p_old)
    i = 0
    
    while abs(p_old-p_new) > tol and i <= maxIterations:
        
        p_old = p_new
        p_new = f(p_old)
        
        i += 1
    
    if maxIterations == 0:   
        print('Max Iterations exceeded (fixed_point_method)') 
        
    return p_new

'''
Using the secant method:

Solves the equation f(x) = 0 on [a, b] with initial
guesses p0 and p1 to given tolerance and max iterations.

Returns zero to within the given tolerance.
'''
def secant_method(f, p0, p1, tol, maxIterations):
    
    i = 2
    
    q0 = f(p0)
    q1 = f(p1)
    p = p0
    
    while abs(q1 - p0) > tol and i <= maxIterations:
        
        p = p1 - q1*(p1-p0)/(q1-q0)
        
        p0 = p1
        q0 = q1
        p1 = p
        q1 = f(p)
        
        if abs(q1 - q0) < 10**(-15):
            return p
        
        i += 1
    
    if(i == maxIterations):
        print('Max Iterations Exceeded (secant_method)')
        
    return p

'''
This performs Aitken acceleration to speed up convergence
of sequences.

This is used in Steffenson's method to quickly find fixed
points.
'''
def aitken(p_n, p_n1, p_n2):
    return p_n - (p_n1-p_n)**2 / (p_n2 - 2*p_n1 + p_n)

'''
Using Steffenson's method:

Solves for the fixed point x = f(x) with an initial
guess p0 to given tolerance and max iterations.

Returns the fixed point.
'''
def steffenson_method(f, p0, tol, maxIterations):
    
    i = 0
    p_n = f(p0)
    p_n1 = f(p_n)
    p_n2 = f(p_n1)
    
    while abs(p_n2 - p_n1) > tol and i <= maxIterations:
        
        p_n = aitken(p_n, p_n1, p_n2)
        p_n1 = f(p_n)
        p_n2 = f(p_n1)
    
    if(i == maxIterations):
        print('Max Iterations Exceeded (steffenson_method)')
    
    return p_n2

