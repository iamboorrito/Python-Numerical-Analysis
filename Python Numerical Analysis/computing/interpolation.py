'''
Created on Aug 10, 2017

@author: evan
'''

from math import cos, pi

"""

Lagrange polynomial in Newton form. Calculated through divided difference
algorithm.

Args:  x_list
       y_list

"""

class interp_poly:
    def __init__(self, x_list, y_list):

        assert len(x_list) == len(y_list)
        
        self.coef = []
        self.x_list = x_list
        self.gen_coef(y_list)

    def deg(self):
        return len(self.coef)-1
    
    def __getitem__(self, x):
        if len(self.coef) == 0:
            return 0.0;
        
        y = 0.0
        
        i = len(self.coef)-1
        # Calculate using nested form
        while i > 0:
            y = (x - self.x_list[i-1])*(y + self.coef[i]);   
            i-=1
        
        return y+self.coef[0] 
    
    def gen_coef(self, y_list):
        
        temp = []
        
        for i in range( len(self.x_list) ):
            temp.append((self.x_list[i], y_list[i]))
            self.coef.append( self.div_diff(temp) );
    
    def div_diff(self, points):
        if len(points) == 1:
            return points[0][1];
        
        n = len(points);
        
        y1 = self.div_diff(points[0:n-1])
        y0 = self.div_diff(points[1:n])
        x1 = points[0][0]
        x0 = points[n-1][0]
                           
        return ( y1 - y0 ) / ( x1 - x0 )

"""

Simple implementation of Chebyshev polynomial with default of n=10 nodes.

Args:  f(x)
       interval [a,b] (as list or tuple with len 2)
       (optional) n = # nodes     

"""
class cheb_poly(interp_poly):
    def __init__(self, f, interval, n=10):

        xy_list = self.gen_nodes(f, interval, n)
        
        # construct normal interpolating polynomial with cheb. nodes
        interp_poly.__init__(self, xy_list[0], xy_list[1])
        
    
    def gen_nodes(self, f, interval, n):
        
        node_list = []
        y_list = []
        a = interval[0]
        b = interval[1]
        
        for i in range(1, n+1):
            node_list.append(0.5*(a+b) + 0.5*(b-a)*cos( (2*i-1)/(2*n)*pi ))
            y_list.append(f(node_list[i-1]))
            
        return (node_list, y_list)

