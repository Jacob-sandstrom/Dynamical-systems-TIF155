# %%
import matplotlib.pyplot as plt
import numpy as np

def dx_dtau(x, r):
    return 1/5 + 7/10/(1 + np.exp(80*(1-x)))-r*x**4

def d2x_dtau2(x, r):
    return 80*np.exp(80*(1-x))*7/10/(1 + np.exp(80*(1-x)))**2-4*r*x**3



r_values = np.linspace(0, 2, 500)
x_range = np.linspace(0.5, 1.5, 5000)

# r_values = np.linspace(0.76, 0.74, 500)
# x_range = np.linspace(0, 1.5, 5000)

stable_fixed_points = []
unstable_fixed_points = []

# n_fixed = 1
# last_n_fixed = 1

for r in r_values:
    derivatives = np.array([dx_dtau(x, r) for x in x_range])
    sign_changes = np.where(np.diff(np.sign(derivatives)))[0]
    
    # last_n_fixed = n_fixed

    # n_fixed = len(sign_changes)
    # if n_fixed != last_n_fixed:
    #     print(f"r: {r:.4f}, Number of fixed points changed: {last_n_fixed} -> {n_fixed}")
    #     print(f"Fixed points at r={r:.4f}: {[x_range[i] for i in sign_changes]}")


    for i in sign_changes:
        

        if d2x_dtau2(x_range[i], r) < 0:
            stable_fixed_points.append((r, x_range[i]))
        else:
            unstable_fixed_points.append((r, x_range[i]))



stable_points = np.array(stable_fixed_points)
unstable_points = np.array(unstable_fixed_points)

plt.figure(figsize=(10, 6))


plt.scatter(stable_points[:, 0], stable_points[:, 1], s=1, color='blue', label='Stable Fixed Points')
plt.scatter(unstable_points[:, 0], unstable_points[:, 1], s=1, color='red', label='Unstable Fixed Points')
plt.title('Bifurcation Diagram')
plt.xlabel('r')
plt.ylabel('x*')

plt.vlines([0.26, 0.75],ymin=0.5, ymax=1.5, label="Saddle-Node Bifurcations", colors="black")
plt.legend()
plt.show()
# %%
