'''
Created on Mar 30, 2017

@author: leifh
'''

# Imports and definitions
from sympy import *
from sympy.stats import  Normal, density
init_printing(use_latex='mathjax')
from IPython.display import display;

x1,x2,y =symbols("x1,x2,y")

# End imports and definitions


# Expectation of a normally distributed variable 
mu1 = symbols("mu1", positive=True)
V1 = symbols("sigma1", positive=True)
X1 = Normal("X", mu1, V1)
D1 = density(X1)(x1)
E1=Integral(x1*D1,(x1,-oo,oo))
display(Eq(E1,E1.doit()))   # use doit to evaluate an unevaluated integral


# Compute the variance analytically
V = Integral((x1-E1.doit())**2*D1, (x1,-oo,oo))
display(Eq(V,V.doit()))


# Expectation of the product of two normally distributed parameters
mu2 = symbols("mu2", positive=True)
V2 = symbols("sigma2", positive=True)
X2 = Normal("X", mu2, V2)
D2=density(X2)(x2)
jpdf=D1*D2
E2=Integral(x1*x2*jpdf,(x2,-oo,oo),(x1,-oo,oo)) 
display(Eq(E2,E2.doit())) # use doit to evaluate an unevaluated integral

    
# Conditional expectation
E_given_x1=Integral(x2*jpdf,(x2,-oo,oo)) 
display(E_given_x1.doit())

from sympy.plotting import plot
mu1_value=1
V1_value=2
mu2_value=2
V2_value=1

Ex=E_given_x1.subs([(V1,V1_value),(mu2,mu2_value),(mu1,mu1_value)]).doit()
_=plot(Ex,(x1,-10,10))

# Compute the conditional variance analytically
V_given_x1=Integral((x2-E_given_x1.doit())**2*jpdf,(x2,-oo,oo))
display(simplify(V_given_x1.doit())) 
Vx=V_given_x1.subs([(mu1,mu1_value),(V1,V1_value),(mu2,mu2_value),(V2,V2_value)]).doit()
display(Vx)
_=plot(Vx,(x1,-10,10))



