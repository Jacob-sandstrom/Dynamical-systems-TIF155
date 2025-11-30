import sympy as sp
from sympy.abc import x, y, z

t, s, x, y, l1, l2 = sp.symbols("t, sigma x y lambda1 lambda2")


A = sp.Matrix([[s+1, 3],[-2, s-1]])

print(A)


print()
l1, l2 = list(A.eigenvals().keys())
print(l1)
print(l2)
print()
v1, v2  = sp.symbols("v1 v2")
v1 = A.eigenvects()[0][2][0]
v2 = A.eigenvects()[1][2][0]

print(v1)
print(v2)
# print(A.eigenvects()[1][2])

u, v, c_1, c_2 = sp.symbols("u v c_1 c_2")



sol = sp.solve([c_1*v1[0] + c_2*v2[0]-u, c_1*v1[1] + c_2*v2[1]-v], [c_1, c_2], dict=True)

c_1 = sol[0][c_1]
c_2 = sol[0][c_2]

print(c_1)
print(c_2)


x = c_1*v1*sp.exp(l1*t) + c_2*v2*sp.exp(l2*t)
print()

print()
print(x[0])
print(x[1])
