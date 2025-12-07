# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import eigvals
from scipy.integrate import solve_ivp

a = 4 / 9
b = 5 / 9
epsilon = 1
I_vals = np.linspace(0, 1, 100) 


def system(point, I, a, b, epsilon):
    x, y = point
    x_dot = (x - x**3 / 3 - y + I) / epsilon
    y_dot = x + a - b * y
    return np.array([x_dot, y_dot])

def system_t(t,point, I, a, b, epsilon):
    x, y = point
    x_dot = (x - x**3 / 3 - y + I) / epsilon
    y_dot = x + a - b * y
    return np.array([x_dot, y_dot])


def jacobian(point, a, b, epsilon):
    x, y = point
    J = np.array([[ (1 - x**2) / epsilon, -1 / epsilon],
                  [1, -b]])
    return J


# %% 
# b

real_parts = np.zeros((len(I_vals), 2))
imag_parts = np.zeros((len(I_vals), 2))

for i in I_vals:
    fp = fsolve(system, [0, 0], args=(i, a, b, epsilon))
    j = jacobian(fp, a, b, epsilon)

    eigenvalues = eigvals(j)
    real_parts[I_vals == i, :] = np.real(eigenvalues)
    imag_parts[I_vals == i, :] = np.imag(eigenvalues)


plt.plot(I_vals, real_parts[:, 0], label=r'Real $\lambda_{1,2}$')
# plt.plot(I_vals, real_parts[:, 1], label=r'Real $\lambda_2$')

plt.plot(I_vals, np.abs(imag_parts[:, 0]), label=r'|Imag $\lambda_{1,2}$|')
# plt.plot(I_vals, np.abs(imag_parts[:, 1]), label=r'|Imag $\lambda_2$|', linestyle='--')

plt.vlines(x=68/405, ymin=-0.15, ymax=0.95, colors="black", label='Bifurcation Point')

plt.xlabel('I')
plt.ylabel('Eigenvalue Parts')
plt.legend()
plt.show()


# %%
# c
a = 4 / 9
b = 5 / 9
epsilon = 1
i_c = 68/405

i_low = i_c*0.7
i_high = i_c*1.3

start_points = np.array([[-0.5, 0], [2, 2], [-2, -2]])

fix, ax = plt.subplots(1,2)

t_vals = np.linspace(0, 100, 10000)
for start in start_points:
    sol = solve_ivp(system_t, [t_vals[0], t_vals[-1]], start, args=(i_low, a, b, epsilon), t_eval=t_vals)
    traj = sol.y
    ax[0].plot(traj[0, :], traj[1, :], label=f'Start: {start}')

    sol = solve_ivp(system_t, [t_vals[0], t_vals[-1]], start, args=(i_high, a, b, epsilon), t_eval=t_vals)
    traj = sol.y
    ax[1].plot(traj[0, :], traj[1, :], label=f'Start: {start}')


ax[0].set_title(f'I = {i_low:.3f} < I_c')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].legend()

ax[1].set_title(f'I = {i_high:.3f} > I_c')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].legend()

ax[0].axis('equal')
ax[1].axis('equal')
plt.tight_layout()
plt.show()



# %%
# d
def x_nullcline(x, I):
    return x - x**3 / 3 + I

def y_nullcline(x, a, b):
    return (x + a) / b

a = 1
b = 1
epsilon = 1/100
i_c = 1-99*np.sqrt(11)/1000

i = 0.3
# i = 1

fp = fsolve(system, [0, 0], args=(i, a, b, epsilon))
print(f'Fixed Point at I={i}: {fp}')

small_perturbation = 0.1
large_perturbation = 0.2

small, large = np.array([fp-[0,small_perturbation], fp-[0,large_perturbation]])

fix, ax = plt.subplots(1)

t_vals = np.linspace(0, 10, 1000)


sol = solve_ivp(system_t, [t_vals[0], t_vals[-1]], small, args=(i, a, b, epsilon), t_eval=t_vals)
traj_s = sol.y
ax.plot(traj_s[0, :], traj_s[1, :], label=f'small perturbation start')

sol = solve_ivp(system_t, [t_vals[0], t_vals[-1]], large, args=(i, a, b, epsilon), t_eval=t_vals)
traj_l = sol.y
ax.plot(traj_l[0, :], traj_l[1, :], label=f'large perturbation start')


x_range = np.linspace(-2, 2, 400)
ax.plot(x_range, x_nullcline(x_range, i), label='x nullcline', linestyle='--')
ax.plot(x_range, y_nullcline(x_range, a, b), label='y nullcline', linestyle='--')
ax.scatter(fp[0], fp[1], color='red', label='Fixed Point')


ax.set_title(f'I = {i:.3f} < I_c')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis('equal')
ax.set_xlim([-2, 2])
ax.set_ylim([-0.75, 1.5])
ax.legend()
plt.tight_layout()
plt.show()


plt.plot(t_vals, traj_s[0, :], label='small perturbation x(t)')
plt.plot(t_vals, traj_l[0, :], label='large perturbation x(t)')
plt.xlabel('Time')
plt.ylabel('x')
plt.legend()
plt.show()

# %%
