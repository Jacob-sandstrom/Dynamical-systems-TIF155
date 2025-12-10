# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def system(t, state, sigma, r, b):
    x, y, z = state
    dxdt = sigma*(y - x)
    dydt = r*x - y - x*z
    dzdt = x*y - b*z

    return dxdt, dydt, dzdt


sigma = 10
r = 28
b = 8 / 3

start = [0.01, 0.01, 0.01]
t_vals = np.linspace(0, 100, 10000)
sol = solve_ivp(system, [t_vals[0], t_vals[-1]], start, args=(sigma, r, b), t_eval=t_vals)
traj = sol.y

fig = plt.figure()
ax = plt.axes(projection='3d')
plot_start = 2000
ax.plot(traj[0, plot_start:], traj[1, plot_start:], traj[2, plot_start:], lw=0.2)
plt.show()