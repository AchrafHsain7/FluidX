import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.LBMSolvers import LBMSolver2D
from core.visualizers import visualize, readmask
from utils.utils import generateLatticesConfig

import torch
import numpy as np
from tqdm import tqdm



if __name__ == "__main__":
    N_X, N_Y = 500, 300
    DEVICE = "cuda"
    PRECISION = torch.float32
    N_DVELOCITIES = 9 #number of discrete velocities
    
    #0 Left, 1: Top, 2: Right, 3: Bottom
    
    


    LATTICES, OPPOSITE_LATTICES, Ci, WEIGHTS, DIRECTIONAL_VELOCITIES = generateLatticesConfig("D2Q9")
    WEIGHTS = torch.tensor(WEIGHTS).to(DEVICE)
    Ci = torch.tensor(Ci, dtype=PRECISION).to(DEVICE)
    LATTICES = torch.tensor(LATTICES).to(DEVICE)
    OPPOSITE_LATTICES = torch.tensor(OPPOSITE_LATTICES).to(DEVICE)

    
    VERTICAL_VELOCITIES = torch.tensor([0, 2, 4]).to(DEVICE)
    HORIZENTAL_VELOCITIES = torch.tensor([0, 1, 3]).to(DEVICE)



    #SIMULATION PARAMETERS
    N_ITERATIONS = 30_000
    RIGHT_VELOCITY = 0.02 #mach
    CYLINDER_RADIUS = N_Y // 9
    REYNOLD_NUMBER = 5000
    MASK = "../../data/models/airplane.jpg"
    

    velcity_profile = torch.zeros((N_X, N_Y, 2)).to(DEVICE)
    velcity_profile[:, :, 0] = RIGHT_VELOCITY
    f = torch.ones((N_X, N_Y, 9)).to(DEVICE)
    X, Y = np.meshgrid(np.arange(N_X), np.arange(N_Y), indexing="ij")
    CYLINDER_CENTER = [N_X // 5, N_Y // 2]
    CYLINDER_RADIUS = N_Y // 7
    N_PLOT = 150
    mask = np.sqrt(
        (X - CYLINDER_CENTER[0])**2 + (Y - CYLINDER_CENTER[1])**2
        ) < CYLINDER_RADIUS
    L = CYLINDER_RADIUS
    mask = torch.tensor(mask).to(DEVICE)

    mask, L = readmask(MASK, N_X, N_Y, ratio_Y=0.3, leftoffset=50)

    kinematic_viscosity = (RIGHT_VELOCITY * L) / REYNOLD_NUMBER
    RELAXATION_TIME = 1.0 / (3 * kinematic_viscosity + 0.5)  # TODO: Check how it is actuallu computed
    CONFINED_MODE = False



    lbm = LBMSolver2D(Ci, WEIGHTS, RELAXATION_TIME, N_DVELOCITIES, torch.ones((N_X, N_Y)).to(DEVICE), 
                      DIRECTIONAL_VELOCITIES, velcity_profile, VERTICAL_VELOCITIES, LATTICES, OPPOSITE_LATTICES, confined=CONFINED_MODE)
    

    for i in tqdm(range(N_ITERATIONS)):
        f = lbm.update(mask)
        if i % N_PLOT == 0:
            visualize(f, X, Y, Ci)
    
