import numpy as np
from sympy import *
import matplotlib.pyplot as plt


def runge_kutta(x, A, dt):
    k1 = (A @ x) * dt
    k2 = (A @ (x + 0.5 * k1)) * dt
    k3 = (A @ (x + 0.5 * k2)) * dt
    k4 = (A @ (x + k3)) * dt
    return x +  (k1 + 2*k2 + 2*k3 + k4) / 6


sigma_val = [-1, 0, 1]
dt = 0.1



for sigma in sigma_val:
    A = np.array([[sigma + 3, 4],
                [-9/4, sigma - 3]])
    # print(A)

    eigval, eigvec = np.linalg.eig(A)

    # print(eigval)
    # print(eigvec)

    # print()
    
    
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x_range = np.linspace(-5, 5, 15)
    y_range = np.linspace(-5, 5, 15)
    
    for x0 in x_range:
        for y0 in y_range:
            x = np.array([x0, y0])
            trajectory_x = [x[0]]
            trajectory_y = [x[1]]
            
            for _ in range(20):
                x = runge_kutta(x, A, dt)
                trajectory_x.append(x[0])
                trajectory_y.append(x[1])
            
            ax.plot(trajectory_x, trajectory_y, alpha=0.6, linewidth=0.8)
            if x0 == x_range[7]:
                if y0 != 0:
                    ax.arrow(trajectory_x[4], trajectory_y[4], trajectory_x[5]-trajectory_x[4], trajectory_y[5]-trajectory_y[4], head_width=0.3, head_length=0.2, fc='red', ec='red', linewidth=2)



    for i in range(2):
        vec = eigvec[:, i]
        if sigma == -1:
            ax.arrow(vec[0]*2, vec[1]*2, vec[0]*0.5*eigval[0], vec[1]*0.5*eigval[0], head_width=0.3, head_length=0.2, fc='red', ec='red', linewidth=2)
            ax.arrow(vec[0]*5, vec[1]*5, vec[0]*0.5*eigval[0], vec[1]*0.5*eigval[0], head_width=0.3, head_length=0.2, fc='red', ec='red', linewidth=2)
            ax.arrow(0, 0, vec[0]*20, vec[1]*20)
            ax.text(-5, 5, "Stable fix point at (0,0)")
        elif sigma == 0:
            ax.text(-5, 5, "Unstable fix points at the whole black line")
            ax.arrow(0, 0, vec[0]*20, vec[1]*20)
        else:
            ax.text(-5, 5, "Unstable fix point at (0,0)")
            ax.arrow(vec[0]*2, vec[1]*2, vec[0]*0.5*eigval[0], vec[1]*0.5*eigval[0], head_width=0.3, head_length=0.2, fc='red', ec='red', linewidth=2)
            ax.arrow(vec[0]*5, vec[1]*5, vec[0]*0.5*eigval[0], vec[1]*0.5*eigval[0], head_width=0.3, head_length=0.2, fc='red', ec='red', linewidth=2)
            ax.arrow(0, 0, vec[0]*20, vec[1]*20)

    ax.scatter(0,0)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'Phase Portrait for σ = {sigma}')
    plt.show()