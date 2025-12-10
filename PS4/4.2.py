# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def system(t, state):
    x, y = state
    dxdt = 1/10*x-y**3-x*y**2-x**2*y-y-x**3
    dydt = x+y/10+x*y**2+x**3-y**3-x**2*y

    return dxdt, dydt



start = [0.01, 0.01]
t_vals = np.linspace(0, 100, 10000)

plot_start = 0

for start in [[0.01, 0.01], [1, 1], [-1, -1], [1, -1], [-1, 1]]:
    sol = solve_ivp(system, [t_vals[0], t_vals[-1]], start, args=(), t_eval=t_vals)
    traj = sol.y
    plt.plot(traj[0, plot_start:], traj[1, plot_start:], lw=0.2)

plt.show()


# %%
# d
def jacobian(X1, X2):
    j11 = 0.1 - X2**2 - 2*X1*X2 - 3*X1**2
    j12 = -3*X2**2 - 2*X1*X2 - X1**2 - 1
    j21 = 1 + X2**2 + 3*X1**2 - 2*X1*X2
    j22 = 0.1 + 2*X1*X2 - 3*X2**2 - X1**2

    return np.array([[j11, j12], [j21, j22]])

def m_system(t, state):
    X1, X2, m11, m12, m21, m22 = state

    dX1_dt, dX2_dt = system(t, [X1, X2])

    m = np.array([[m11, m12], [m21, m22]])
    J = jacobian(X1, X2)
    dm_dt = J @ m

    return dX1_dt, dX2_dt, dm_dt[0,0], dm_dt[0,1], dm_dt[1,0], dm_dt[1,1]


T = 2*np.pi/(1+1*1/10)
print(T)
start = [np.sqrt(0.1), 0, 1, 0, 0, 1]
t_vals = np.linspace(0, T, 10000)

plot_start = 0

sol = solve_ivp(m_system, [t_vals[0], t_vals[-1]], start, args=(), t_eval=t_vals)
traj = sol.y
plt.plot(t_vals, traj[0, plot_start:], label=r'$X_{1}$')
plt.plot(t_vals, traj[1, plot_start:], label=r'$X_{2}$')
plt.plot(t_vals, traj[2, plot_start:], label=r'$M_{11}$')
plt.plot(t_vals, traj[3, plot_start:], label=r'$M_{12}$')
plt.plot(t_vals, traj[4, plot_start:], label=r'$M_{21}$')
plt.plot(t_vals, traj[5, plot_start:], label=r'$M_{22}$')
plt.legend()

plt.show()
m = np.array([[traj[2, -1], traj[3, -1]], [traj[4, -1], traj[5, -1]]])
print(m)
# %%

eigvals = np.linalg.eigvals(m)
stability_exponent = np.log(np.abs(eigvals))/T
print(eigvals)
print(stability_exponent)
# %%
