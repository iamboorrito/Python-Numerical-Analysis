'''
Created on July 21, 2017

@author: whowasfon
'''

from numpy import zeros
from math import sqrt

'''
Trapezoid rule for one interval [a, b]

Args: Continuous function f(x)
      Interval [a, b]

'''
def trapezoid(f, a, b):
    return 0.5*(b-a)(f(a) + f(b))

'''
Simpson's rule for one interval [a, b]

Args: Continuous function f(x)
      Interval [a, b]

'''
def simpson(f, a, b):
    return (b-a)*(f(a) + 4*f((a+b)/2) + f(b))/12.0

'''
Composite Trapezoid rule for n subintervals

Args: Continuous function f(x)
      Interval [a, b]
      Number of subintervals = n

'''
def composite_trapezoid(f, a, b, n):
    # Had problems with python assigning h as int
    h = (b-a)/(0.0+n)
    
    S0 = f(a) + f(b)
    
    S1 = 0
    
    for i in range(1, n):
        S1 += f(a + i*h)
    
    return h*(S0 + 2*S1)/2

'''
Composite Simpson's rule for n subintervals. 

Args: Continuous function f(x)
      Interval [a, b]
      Number of subintervals = n

Note, n should be even for effectiveness. There is no guarantee
that composite Simpson's will give any accuracy with odd n.

'''
def composite_simpson(f, a, b, n):
    
    h = (b-a)/(0.0+n)

    S0 = f(a) + f(b)
    
    S1 = 0
    S2 = 0
    
    for i in range(1, int(n/2)):
        S1 += f(a+2*i*h)
        
    for i in range(1, int(n/2)+1):
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

Romberg Integration

Args: Continuous function f(x)
      Interval [a, b]
      Romberg Table height = n

Calculates up to R(n, n) with order of precision given
by O(h**2n). Integrates f from a to b with n rows and columns.

Returns a numpy array of size n*(n+1)/2 instead of a table 
of size n*n. To get table form, pass the result to the
function create_romberg_table(romberg_table).

To access R(k, j), where indices start at 0, use:

    R(k, j) = k*(ROWS-1) + k*(COLS-1) + 1
    
Example Use:
    
from math import cos
romb_array = romberg_method(cos, 0, 1, 3)
romb_table = create_romberg_table(romb_array)

print(romb_table)
print(romb_table[2,2])
print(romb_array[2*1+2*1 + 1])
    
'''
def romberg_method(f, a, b, n):
    
    table = zeros((int(n*(n+1)/2))) 
    i = 0
    
    for row in range(0, n):
        for col in range(0, row+1):
            table[i] = romberg_iteration(row+1, col+1, f, a, b)
            i += 1
    
    return table

'''
Returns an n x n table given the output of romberg_method.

Args: Output from romberg_method(...)

'''
def create_romberg_table(romb_table):
    
    
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
