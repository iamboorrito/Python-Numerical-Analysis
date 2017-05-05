'''
Created on May 2, 2017

@author: Evan Burton
'''

from numpy import zeros
from math import sqrt

'''
Trapezoid rule for one interval [a, b]
'''
def trapezoid(f, a, b, h):
    return 0.5*h(f(a) + f(b))

'''
Simpson's rule for one interval [a, b]
'''
def simpson(f, a, b, h):
    return h*(f(a) + 4*f((a+b)/2) + f(b))/6

'''
Composite Trapezoid rule for n subintervals
'''
def composite_trapezoid(f, a, b, n):
    
    h = (b-a)/(0.0+n)
    
    S0 = f(a) + f(b)
    
    S1 = 0
    
    for i in range(1, n):
        S1 += f(a + i*h)
    
    return h*(S0 + 2*S1)/2

'''
Composite Simpson's rule for n subintervals. 

Note, n should be even for effectiveness

'''
def composite_simpson(f, a, b, n):
    
    h = (b-a)/(0.0+n)

    S0 = f(a) + f(b)
    
    S1 = 0
    S2 = 0
    
    for i in range(1, n/2):
        S1 += f(a+2*i*h)
        
    for i in range(1, n/2+1):
        S2 += f(a+(2*i-1)*h)
    
    return h*(S0 + 2*S1 + 4*S2)/3

'''

Calculates R(k, j) for the Romberg integration scheme.
This is used in romberg_method.

'''
def romberg_iteration(k, j, f, a, b):
    if(j == 1):
        return composite_trapezoid(f, a, b, 2**(k-1))
    else:
        R1 = romberg_iteration(k, j-1, f, a, b)
        R2 = romberg_iteration(k-1, j-1, f, a, b)
        
        return R1 + (R2 - R1)/(4**(j-1)-1)

'''
Calculates up to R(n, n) with order of precision given
by O(h**2n).

Returns a numpy array of size n*(n+1)/2 instead of a table 
of size n*n. To get table form, pass the result to the
function create_table(romberg_table).

To access R(k, j), where indices start at 0, use:

    R(k, j) = k*ROWS + k*COLS + 1
    
'''
def romberg_method(f, a, b, n):
    
    table = zeros((n*(n+1)/2))
    
    '''
    
    Save table space in memory by using flat array:
    n*(n+1)/2 elements
    0  1  2  3  4  5
    
    Instead of wasting spaces that are never filled in:
    0
    1  2
    3  4  5 

    R[2, 2] = 2*rows + 2*cols + 1
    
    table[0, 0] = table[0]
    table[1, 0] = table[1]
    table[1, 1] = table[2]
    table[2, 0] = table[3]
    table[2, 1] = table[4]
    table[2, 2] = table[5]
    
    k = 0

    for row in range(0, n):
        for col in range(0, row+1):
            table[k] = romberg_iteration(row+1, col+1, f, a, b)
            k += 1
    '''
    
    i = 0
    
    for row in range(0, n):
        for col in range(0, row+1):
            table[i] = romberg_iteration(row+1, col+1, f, a, b)
            i += 1
    
    return table
'''
Returns an nxn table given the output of romberg_method.
'''
def create_table(romb_table):
    
    
    '''
    n(n+1)/2 = L
    
    n^2 + n - 2L = 0
    
    => n = (sqrt(8L+1)-1)/2
    
    by quadratic formula
    
    '''
    
    n = romb_table.shape[0]
    #always results in an int anyway if given a romberg array
    n = int((sqrt(8*n+1)-1)/2)
    
    table = zeros((n, n))
    
    j = 0
    
    for row in range(0, n):
        for col in range(0, row+1):
            table[row, col] = romb_table[j]
            j += 1
            
    return table

'''
from math import cos

rombergArray = romberg_method(cos, 0, 1, 3)
romb_table = create_table(rombergArray)

print rombergArray
print romb_table

print rombergArray[1*3 + 3*3 + 1]
'''