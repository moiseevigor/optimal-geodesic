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

def odesyst(t, inp):
    """
    Hamiltonian system that plots geodesic trajectories for S^2
    \dot x = h_1 cos(th) -> 0
    \dot y = h_1 sin(th) -> 1
    \dot th = h_2        -> 2

    H = h_1^2 + h_2^2 = C/2 => gives geodesic parametrization
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


r = 3
num_points = 500
alpha = np.linspace(-2*r, 2*r, int(num_points)+1)
h_10 = 2*np.sin(alpha)
h_20 = 2*np.cos(alpha)
h_30 = 1

initial_state =  torch.tensor([
    h_10,                       # h1
    h_20,                       # h2
    torch.ones(len(h_10))*h_30, # h3
    torch.zeros(len(h_10)),     # x
    torch.zeros(len(h_10)),     # y 
    torch.zeros(len(h_10)),     # th
]).transpose(1,0)

# num of steps
N_steps = 501
t = np.linspace(0, 2, N_steps)
sol = odeint(
    odesyst, 
    initial_state,
    torch.Tensor(t),
    rtol=1e-7,
    atol=1e-9,
    method=backend
)
# import pdb; pdb.set_trace()

fig = plt.figure()
# plt.gca().set_aspect('equal', adjustable='box')
ax = plt.axes(projection='3d')
# ax = plt.axes(projection='rectilinear')

ax.plot_surface(sol[:,:, 3], sol[:,:, 4], sol[:,:, 5])
# for it in range(0,len(sol[0,:, 3])):
    # plt.plot(sol[:,it, 3], sol[:,it, 4])
    # ax.plot3D(sol[:,it, 3], sol[:,it, 4], sol[:,it, 5])

plt.show()
