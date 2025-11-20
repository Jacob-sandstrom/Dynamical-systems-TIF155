#%%
import numpy as np
import matplotlib.pyplot as plt


def f_prime(x,h,r):
    return r - 2*x

def x_fun(h,r):
    return (r + np.sqrt(r**2 + 4*h)) / 2, (r - np.sqrt(r**2 + 4*h)) / 2

def h_r(r):
    return -r**2 / 4


r_values = np.linspace(-2, 2, 100)

h_curve = h_r(r_values)
x_curve = r_values / 2, r_values / 2

h_values = np.linspace(-1, 1, 100)


stable = []
unstable = []

points = []

for r in r_values:
    for h in h_values:

        x1, x2 = x_fun(h, r)

        for x in (x1, x2):
            if not np.isnan(x):
                if f_prime(x, h, r) < 0:
                    stable.append((r, h, x))
                else:
                    unstable.append((r, h, x))

stable = np.array(stable)
unstable = np.array(unstable)


plt.scatter(stable[:,0], stable[:,1], s=1, color='blue', label='2 fixed, 1 stable, 1 unstable')
plt.scatter([], [], color='w', label='0 fixed')  
plt.plot(r_values, h_curve, color='black', linewidth=4, label='Bifurcation Curve')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(stable[:, 0], stable[:, 1], stable[:, 2], c='blue', s=1, label='Stable', alpha=0.4)
ax.scatter(unstable[:, 0], unstable[:, 1], unstable[:, 2], c='red', s=1, label='Unstable', alpha=0.4)
ax.plot(r_values, h_curve, x_curve[0], color='black', linewidth=4, label='Bifurcation Curve', zorder=10)

ax.set_xlabel('r')
ax.set_ylabel('h')
ax.set_zlabel('x*')
ax.set_title('Bifurcation Surface (r, h, x*)')
ax.legend()
plt.show()
# %%