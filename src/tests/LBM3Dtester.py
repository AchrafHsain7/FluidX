import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.LBMSolvers import LBMSolver3D
from core.visualizers import visualize, readmask
from utils.utils import generateLatticesConfig

import torch
import numpy as np
from tqdm import tqdm
import cProfile
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



if __name__ == "__main__":
    N_X, N_Y, N_Z = 500, 200, 200
    DEVICE = "cuda"
    PRECISION = torch.float32
    N_DVELOCITIES = 19 #number of discrete velocities



    LATTICES, OPPOSITE_LATTICES, Ci, WEIGHTS = generateLatticesConfig("D3Q19")

    WEIGHTS = torch.tensor(WEIGHTS).to(DEVICE)
    Ci = torch.tensor(Ci, dtype=PRECISION).to(DEVICE)
    LATTICES = torch.tensor(LATTICES).to(DEVICE)
    OPPOSITE_LATTICES = torch.tensor(OPPOSITE_LATTICES).to(DEVICE)



    #0 Left, 1: Top, 2: Right, 3: Bottom
    DIRECTIONAL_VELOCITIES = torch.tensor([
        [3, 6, 7],#left  where X lattice coordinate == -1
        [2, 5, 6],#top
        [1, 5, 8],#right
        [4, 7, 8]#bottom
    ]).to(DEVICE)
    VERTICAL_VELOCITIES = torch.tensor([0, 2, 4]).to(DEVICE)
    HORIZENTAL_VELOCITIES = torch.tensor([0, 1, 3]).to(DEVICE)



    #SIMULATION PARAMETERS
    RIGHT_VELOCITY = 0.05 #mach
    CYLINDER_RADIUS = N_Y // 9
    REYNOLD_NUMBER = 1200
    MASK = "../../data/models/airplane.jpg"
    

    velcity_profile = torch.zeros((N_X, N_Y, N_Z, 3)).to(DEVICE)
    velcity_profile[:, :, :, 0] = RIGHT_VELOCITY


    f = torch.ones((N_X, N_Y, N_DVELOCITIES)).to(DEVICE)
    X, Y, Z = np.meshgrid(np.arange(N_X), np.arange(N_Y), np.arange(N_Z), indexing="ij")

    CYLINDER_CENTER = [N_X // 5, N_Y // 2, N_Z//2]
    CYLINDER_RADIUS = N_Y // 7
    N_PLOT = 150

    mask = np.sqrt(
        (X - CYLINDER_CENTER[0])**2 + (Y - CYLINDER_CENTER[1])**2 + (Z - CYLINDER_CENTER[2])**2
        ) < CYLINDER_RADIUS
    

    #plotting 3D
    # import pyvista as pv
    # # Create a PyVista grid and visualize it
    # grid = pv.ImageData()
    # grid.dimensions = (N_X+1, N_Y+1, N_Z+1)
    # grid.cell_data['values'] = mask.flatten(order='F')  # Flatten the array in Fortran order
    # grid = grid.threshold(0.5)  # Only keep voxels with True values
    # grid.plot()

    
    
    L = CYLINDER_RADIUS
    mask = torch.tensor(mask).to(DEVICE)

    # mask, L = readmask(MASK, N_X, N_Y, ratio_Y=0.3, leftoffset=50)

    kinematic_viscosity = (RIGHT_VELOCITY * L) / REYNOLD_NUMBER
    RELAXATION_TIME = max(1.0 / (3 * kinematic_viscosity + 0.5), 0.5)  # TODO: Check how it is actuallu computed
    print(RELAXATION_TIME)



    lbm = LBMSolver3D(Ci, WEIGHTS, RELAXATION_TIME, N_DVELOCITIES, torch.ones((N_X, N_Y, N_Z)).to(DEVICE), 
                      DIRECTIONAL_VELOCITIES, velcity_profile, VERTICAL_VELOCITIES, LATTICES, OPPOSITE_LATTICES)
    

    # for i in tqdm(range(15000)):
    #     f = lbm.update(mask)
    #     if i % N_PLOT == 0:
    #         visualize(f, X, Y, Ci)

    lbm.propagate(lbm.discreteFluid)
    
    # cProfile.run("visualize(f, X, Y, Ci)", sort="time")
