import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.LBMSolvers import LBMSolver2D
from utils.utils import generateLatticesConfig, computeDensity
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



if __name__ == "__main__":

    #GENERAL
    N_X, N_Y = 200, 200
    DEVICE = "cuda"
    PRECISION = torch.float32
    N_DVELOCITIES = 9
    LATTICES, OPPOSITE_LATTICES, Ci, WEIGHTS, DIRECTIONAL_VELOCITIES = generateLatticesConfig("D2Q9")
    WEIGHTS = torch.tensor(WEIGHTS).to(DEVICE)
    Ci = torch.tensor(Ci, dtype=PRECISION).to(DEVICE)
    LATTICES = torch.tensor(LATTICES).to(DEVICE)
    OPPOSITE_LATTICES = torch.tensor(OPPOSITE_LATTICES).to(DEVICE)
    #SIMULATION PARAMETERS
    N_ITERATIONS = 200
    N_PLOT = 2
    RIGHT_VELOCITY = 0.0005
    CYLINDER_RADIUS = N_Y // 9
    REYNOLD_NUMBER = 1
    CONFINED_MODE = True
    X, Y = np.meshgrid(np.arange(N_X), np.arange(N_Y), indexing="ij")
    mask = torch.zeros(X.shape, dtype=bool).to(DEVICE) #no mask
    L = 1

    #INITIAL DENSITY
    INITIAL_DENSITY = torch.ones((N_X, N_Y))
    CYLINDER_CENTER = [N_X //2,  N_Y//2]
    CYLINDER_RADIUS = N_X // 10
    DENSITY_PULSE = (np.sqrt(((X - CYLINDER_CENTER[0])**2 + (Y - CYLINDER_CENTER[1])**2)) < CYLINDER_RADIUS).astype(bool)

    rho0 = 1.0
    delta_rho = 0.1
    gaussian = np.exp(-(((X - CYLINDER_CENTER[0])**2 + (Y - CYLINDER_CENTER[1])**2) / (2 * CYLINDER_RADIUS**2)))
    gaussian = rho0 + delta_rho * gaussian
    INITIAL_DENSITY[DENSITY_PULSE] = torch.tensor(gaussian[DENSITY_PULSE], dtype=PRECISION)
    INITIAL_DENSITY = INITIAL_DENSITY.to(DEVICE)
    print(INITIAL_DENSITY.mean())
    # print(INITIAL_DENSITY)

    #SOLVER
    lbm = LBMSolver2D(Ci, WEIGHTS, REYNOLD_NUMBER, N_DVELOCITIES, INITIAL_DENSITY, 
                      DIRECTIONAL_VELOCITIES, RIGHT_VELOCITY, LATTICES, 
                      OPPOSITE_LATTICES, L,  confined=CONFINED_MODE, device=DEVICE)
    
    # plt.imshow(INITIAL_DENSITY[:, :, N_Z//2].cpu(), cmap="coolwarm")
    # plt.pause(0.01)
    # plt.show()
    
    #main loop
    for i in tqdm(range(N_ITERATIONS)):
        d = lbm.benchmark().cpu()
        plt.subplot(1, 2, 1)
        plt.imshow(d.T, cmap="coolwarm")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.subplot(1, 2, 2)
        plt.plot(d.mean(dim=-1), c="r")
        # plt.ylim([0.999, 1.001])
        plt.pause(0.01)
        plt.clf()
        # plt.show()