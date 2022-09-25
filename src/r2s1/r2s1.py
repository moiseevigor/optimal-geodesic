import torch
from torch import nn, optim
import math
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time

# import sys
# sys.path.append("/app/lib")
# from lib.viz import plot_trajectories, plot_wavefronts

# from scipy.integrate import odeint
from torchdiffeq import odeint
# "explicit_adams", "fixed_adams", "adams", "tsit5", "dopri5", "bosh3", "euler", "midpoint", "rk4", "adaptive_heun", "dopri8"
backend = 'euler'

# from functions.blocks import mins, colors, loss
# from functions.twomin import mins, colors, loss
# from functions.blocks_rastrigin import mins, colors, loss
# from functions.rastrigin import mins, colors, loss
# from functions.threemins import mins, colors, loss

def plot_trajectories(steps, mins, colors, style='-'):
    list_idx = set()
    complete_list_idx = set()

    for idx_min, min in enumerate(mins):
        # print(idx_min, min)
        for idx, is_min in enumerate(np.linalg.norm(steps[-1, :, :] - min, axis=1)<0.05):
            complete_list_idx.add(idx)
            if is_min:
                list_idx.add(idx)
                plt.plot(steps[0, idx, 0], steps[0, idx, 1], 'o', color=colors[idx_min], alpha=0.8)
                plt.plot(steps[:, idx, 0], steps[:, idx, 1], style, color=colors[idx_min], alpha=0.8)
                if style == '-':
                    plt.plot(steps[:, idx, 0], steps[:, idx, 1], '*', color=colors[idx_min], alpha=0.8)

    for idx in (complete_list_idx-list_idx):
        plt.plot(steps[0, idx, 0], steps[0, idx, 1], 'o', color='g', alpha=0.8)
        plt.plot(steps[:, idx, 0], steps[:, idx, 1], style, color='g', alpha=0.8)

    mmins = np.concatenate([mins, [mins[0,:]]], axis=0)
    # plt.plot(mmins[:,0], mmins[:,1], '-', color='k', alpha=1, markersize=18)
    for idx_min, min in enumerate(mins):
        plt.plot(min[0], min[1], '*', color='g', alpha=1, markersize=24)
        plt.plot(min[0], min[1], '*', color=colors[idx_min], alpha=1, markersize=10)

# num of steps
N_steps = 501
momentum = 0.0
alpha = 0.001

def odesyst(t, inp):
    """
    Hamiltonian system that proves extremal trajectories for R^2 x S^1
    \dot h_1 = -h_2*h_3
    \dot h_2 =  h_1*h_3
    \dot h_3 =  h_2*h_3

    \dot x = h_1 cos(th)
    \dot y = h_1 sin(th)
    \dot th = h_2

    """
    return torch.Tensor([
        -inp[1]*inp[2],
         inp[0]*inp[2],
         inp[1]*inp[2],

         inp[0]*torch.cos(inp[5]),
         inp[0]*torch.sin(inp[5]),
         inp[1]
    ])

# r = 5
# num_points = 20
# X = np.linspace(-r, r, int(num_points)+1)
# Y = np.linspace(-r, r, int(num_points)+1)

t = np.linspace(0, 5, N_steps)
initial_state = [1.0,1.0,1.0,0.1,1.0,1.0]

# for x in X:
#     for y in Y:
#         if np.abs(x)>4.9 or np.abs(y)>4.9:
#             print([x,y])
#             initial_state.append([x,y])

initial_state = torch.Tensor(initial_state).requires_grad_(True)

sol = odeint(
    odesyst, 
    initial_state,
    torch.Tensor(t),
    rtol=1e-7,
    atol=1e-9,
    method=backend
)
sol = sol.detach().numpy()
# plot_trajectories(sol, mins, colors, '-')

print(sol)
plt.plot(sol[:, 3], sol[:,4])
plt.show()
