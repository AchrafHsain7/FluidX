import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.LBMSolvers import LBMSolver3D
from core.visualizers import  FluidVisualizer, animate_4d_data
from utils.utils import generateLatticesConfig, computeMacroVelocity, computeDensity, voxel3d

import torch
import numpy as np
from tqdm import tqdm
import cProfile
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



if __name__ == "__main__":
    N_X, N_Y, N_Z = 200, 50, 50
    DEVICE = "cuda"
    PRECISION = torch.float32
    N_DVELOCITIES = 19 #number of discrete velocities



    LATTICES, OPPOSITE_LATTICES, Ci, WEIGHTS, DIRECTIONAL_VELOCITIES = generateLatticesConfig("D3Q19")

    WEIGHTS = torch.tensor(WEIGHTS).to(DEVICE)
    Ci = torch.tensor(Ci, dtype=PRECISION).to(DEVICE)
    LATTICES = torch.tensor(LATTICES).to(DEVICE)
    OPPOSITE_LATTICES = torch.tensor(OPPOSITE_LATTICES).to(DEVICE)


    VERTICAL_VELOCITIES = torch.tensor([0, 5, 6]).to(DEVICE)
    HORIZENTAL_VELOCITIES = torch.tensor([0, 1, 2]).to(DEVICE)



    #SIMULATION PARAMETERS
    RIGHT_VELOCITY = 0.05 #mach
    CYLINDER_RADIUS = N_Y // 5
    REYNOLD_NUMBER = 300
    MASK = "../../data/models/formula1.glb"
    

    velcity_profile = torch.zeros((N_X, N_Y, N_Z, 3)).to(DEVICE)
    velcity_profile[:, :, :, 0] = RIGHT_VELOCITY

    X, Y, Z = np.meshgrid(np.arange(N_X), np.arange(N_Y), np.arange(N_Z), indexing="ij")

    CYLINDER_CENTER = [N_X //5,  N_Y//2, N_Z//2]
    CYLINDER_RADIUS = N_Y // 5
    N_PLOT = 100

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
    L = 50
    mask = voxel3d(MASK, model_dims=L, space_dims=(N_X, N_Y, N_Z), x_shiftf=-30)
    kinematic_viscosity = (RIGHT_VELOCITY * L) / REYNOLD_NUMBER
    RELAXATION_TIME = max(1.0 / (3 * kinematic_viscosity + 0.5), 0.5)

    mask = torch.tensor(mask).to(DEVICE)

      
    print(RELAXATION_TIME)



    lbm = LBMSolver3D(Ci, WEIGHTS, RELAXATION_TIME, N_DVELOCITIES, torch.ones((N_X, N_Y, N_Z)).to(DEVICE), 
                      DIRECTIONAL_VELOCITIES, velcity_profile, VERTICAL_VELOCITIES, LATTICES, OPPOSITE_LATTICES)
    
    vis = FluidVisualizer(X, Y, Z, lbm.discreteFluid, Ci, resolution=1)
    


    animation = []

    for i in tqdm(range(10000)):

        f = lbm.update(mask)
        if i % N_PLOT == 0:
            density = computeDensity(f)
            macroVelocities = computeMacroVelocity(f, density, Ci)
            velocityMagnitude = torch.linalg.norm(macroVelocities, axis=-1, ord=2)
            animation.append(velocityMagnitude.cpu().numpy())
    
    animation = np.array(animation)
    animate_4d_data(animation, "simulation.mp4", fps=10, cmap="jet", opacity=0.2)

