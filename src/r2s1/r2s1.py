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

# num of steps
N_steps = 501
momentum = 0.0
alpha = 0.001

def odesyst(t, inp):
    """
    Hamiltonian system that proves extremal trajectories for R^2 x S^1
    \dot h_1 = -h_2*h_3 -> 0
    \dot h_2 =  h_1*h_3 -> 1
    \dot h_3 =  h_2*h_3 -> 2

    \dot x = h_1 cos(th) -> 3
    \dot y = h_1 sin(th) -> 4
    \dot th = h_2        -> 5
    """
    out = ([
        -inp[:,1]*inp[:,2],
         inp[:,0]*inp[:,2],
         inp[:,1]*inp[:,2],
         inp[:,0]*torch.cos(inp[:,5]),
         inp[:,0]*torch.sin(inp[:,5]),
         inp[:,1]
    ])

    return torch.reshape(torch.cat(out), (6,len(inp))).transpose(1,0)


# r = 5
# num_points = 20
# X = np.linspace(-r, r, int(num_points)+1)
# Y = np.linspace(-r, r, int(num_points)+1)

t = np.linspace(0, 5, N_steps)
# initial_state = [1.0,1.0,1.0,0.1,1.0,1.0]
# initial_state = [[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0],]
initial_state = [
    [1.0,1.0,1.0,0.1,1.0,1.0], # point 0
    [2.0,2.0,2.0,0.2,2.0,2.0], # point 1
    [3.0,3.0,3.0,0.3,3.0,3.0], # point 2
    [0.1,0.2,0.3,0.3,0.1,0.1], # point 2
]

# for x in X:
#     for y in Y:
#         if np.abs(x)>4.9 or np.abs(y)>4.9:
#             print([x,y])
#             initial_state.append([x,y])

initial_state = torch.Tensor(initial_state) #.requires_grad_(True)

sol = odeint(
    odesyst, 
    initial_state,
    torch.Tensor(t),
    rtol=1e-7,
    atol=1e-9,
    method=backend
)
# import pdb; pdb.set_trace()
# sol = sol.detach().numpy()

for it in range(0,4):
    plt.plot(sol[:,it, 3], sol[:,it, 4])
plt.show()
