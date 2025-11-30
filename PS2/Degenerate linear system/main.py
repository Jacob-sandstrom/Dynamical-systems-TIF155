import numpy as np
from sympy import *

sigma, x, y = symbols("sigma, x, y")

A = Matrix([[sigma+3, 4],[-9/4, sigma-3]])
A = Matrix([[2+3, 4],[-9/4, 2-3]])



print(A.eigenvals())