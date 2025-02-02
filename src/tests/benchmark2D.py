import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.LBMSolvers import LBMSolver2D, LBMSolver3D
from utils.utils import generateLatticesConfig
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



if __name__ == "__main__":

    #GENERAL
    N_X, N_Y, N_Z = 50, 50, 50
    DEVICE = "cuda"
    PRECISION = torch.float32
    N_DVELOCITIES = 19 
    LATTICES, OPPOSITE_LATTICES, Ci, WEIGHTS, DIRECTIONAL_VELOCITIES = generateLatticesConfig("D3Q19")
    WEIGHTS = torch.tensor(WEIGHTS).to(DEVICE)
    Ci = torch.tensor(Ci, dtype=PRECISION).to(DEVICE)
    LATTICES = torch.tensor(LATTICES).to(DEVICE)
    OPPOSITE_LATTICES = torch.tensor(OPPOSITE_LATTICES).to(DEVICE)
    #SIMULATION PARAMETERS
    N_ITERATIONS = 1000
    N_PLOT = 2
    RIGHT_VELOCITY = 0.05
    CYLINDER_RADIUS = N_Y // 9
    REYNOLD_NUMBER = 100
    CONFINED_MODE = False
    X, Y, Z = np.meshgrid(np.arange(N_X), np.arange(N_Y), np.arange(N_Z), indexing="ij")
    mask = torch.zeros(X.shape, dtype=bool).to(DEVICE) #no mask
    L = 1

    #INITIAL DENSITY
    INITIAL_DENSITY = torch.zeros((N_X, N_Y, N_Z))
    CYLINDER_CENTER = [N_X //2,  N_Y//2, N_Z//2]
    CYLINDER_RADIUS = N_X // 10
    DENSITY_PULSE = np.sqrt(((X - CYLINDER_CENTER[0])**2 + (Y - CYLINDER_CENTER[1])**2 + (Z-CYLINDER_CENTER[2])**2) < CYLINDER_RADIUS).astype(bool) 
    INITIAL_DENSITY[DENSITY_PULSE] = 1.0
    INITIAL_DENSITY = INITIAL_DENSITY.to(DEVICE)

    print(INITIAL_DENSITY.mean())

    #SOLVER
    lbm = LBMSolver3D(Ci, WEIGHTS, REYNOLD_NUMBER, N_DVELOCITIES, INITIAL_DENSITY, 
                      DIRECTIONAL_VELOCITIES, RIGHT_VELOCITY, LATTICES, 
                      OPPOSITE_LATTICES, L,  confined=CONFINED_MODE)
    
    #main loop
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(111, projection="3d")
    for i in tqdm(range(N_ITERATIONS)):
        d = lbm.benchmark().cpu()
        # plt.imshow(d.cpu(), cmap="coolwarm")
        if i % N_PLOT:
            ax.scatter(X[::2, ::2, ::2], Y[::2, ::2, ::2], Z[::2, ::2, ::2], c=d[::2, ::2, ::2], cmap="coolwarm", alpha=0.3)
            plt.pause(0.001)
            plt.cla()