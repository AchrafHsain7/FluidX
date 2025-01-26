import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.LBMSolvers import LBMSolver2D
from core.visualizers import visualize, readmask

import torch
import numpy as np
from tqdm import tqdm
import cProfile



if __name__ == "__main__":
    N_X, N_Y = 400, 100
    DEVICE = "cuda"
    PRECISION = torch.float32
    N_DVELOCITIES = 9 #number of discrete velocities
    WEIGHTS = torch.tensor([4, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25]).to(DEVICE) / 9
    LATTICES = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
    OPPOSITE_LATTICES = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6])
    #0 Left, 1: Top, 2: Right, 3: Bottom
    DIRECTIONAL_VELOCITIES = torch.tensor([
        [3, 6, 7],#left
        [2, 5, 6],#top
        [1, 5, 8],#right
        [4, 7, 8]#bottom
    ]).to(DEVICE)
    VERTICAL_VELOCITIES = torch.tensor([0, 2, 4]).to(DEVICE)
    HORIZENTAL_VELOCITIES = torch.tensor([0, 1, 3]).to(DEVICE)



    #SIMULATION PARAMETERS
    RIGHT_VELOCITY = 0.1 #mach
    CYLINDER_RADIUS = N_Y // 9
    REYNOLD_NUMBER = 275
    kinematic_viscosity = (RIGHT_VELOCITY * CYLINDER_RADIUS) / REYNOLD_NUMBER
    RELAXATION_TIME = 1.0 / (3 * kinematic_viscosity + 0.5)  # TODO: Check how it is actuallu computed
    MASK = "../../data/models/airfoil.jpg"
    
    Ci = torch.tensor([
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1]
    ], dtype=PRECISION).to(DEVICE)

    velcity_profile = torch.zeros((N_X, N_Y, 2)).to(DEVICE)
    velcity_profile[:, :, 0] = RIGHT_VELOCITY
    f = torch.ones((N_X, N_Y, 9)).to(DEVICE)
    X, Y = np.meshgrid(np.arange(N_X), np.arange(N_Y), indexing="ij")
    CYLINDER_CENTER = [N_X // 5, N_Y // 2]
    CYLINDER_RADIUS = N_Y // 9
    N_PLOT = 150
    mask = np.sqrt(
        (X - CYLINDER_CENTER[0])**2 + (Y - CYLINDER_CENTER[1])**2
        ) < CYLINDER_RADIUS
    mask = torch.tensor(mask).to(DEVICE)

    mask = readmask(MASK, N_X, N_Y, ratio_Y=0.7)



    lbm = LBMSolver2D(Ci, WEIGHTS, RELAXATION_TIME, N_DVELOCITIES, torch.ones((N_X, N_Y)).to(DEVICE), 
                      DIRECTIONAL_VELOCITIES, velcity_profile, VERTICAL_VELOCITIES, LATTICES, OPPOSITE_LATTICES)
    

    for i in tqdm(range(15000)):
        f = lbm.update(mask)
        if i % N_PLOT == 0:
            visualize(f, X, Y, Ci)
    
    # cProfile.run("visualize(f, X, Y, Ci)", sort="time")
