import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.LBMSolvers import LBMSolver3D
from core.visualizers import  FluidVisualizer3D
from utils.utils import *
from core.maskReaders import MaskLoader3D
import torch
import numpy as np
from tqdm import tqdm
from mpl_toolkits import mplot3d
import cmasher as cmr



if __name__ == "__main__":

    N_X, N_Y, N_Z = 200, 50, 50
    DEVICE = "cuda"
    PRECISION = torch.float32
    N_DVELOCITIES = 19 #number of discrete velocities
    N_PLOT = 25

    LATTICES, OPPOSITE_LATTICES, Ci, WEIGHTS, DIRECTIONAL_VELOCITIES = generateLatticesConfig("D3Q19")

    WEIGHTS = torch.tensor(WEIGHTS).to(DEVICE)
    Ci = torch.tensor(Ci, dtype=PRECISION).to(DEVICE)
    LATTICES = torch.tensor(LATTICES).to(DEVICE)
    OPPOSITE_LATTICES = torch.tensor(OPPOSITE_LATTICES).to(DEVICE)


    #SIMULATION PARAMETERS
    RIGHT_VELOCITY = 0.05 #mach
    CYLINDER_RADIUS = N_Y // 5
    CYLINDER_HEIGH = N_Y // 4
    REYNOLD_NUMBER = 200
    MASK = "../../data/models/airplane.glb"
    L = 50
    CONFINED = False
    X, Y, Z = np.meshgrid(np.arange(N_X), np.arange(N_Y), np.arange(N_Z), indexing="ij")
    maskLoader = MaskLoader3D(MASK, L, (N_X, N_Y, N_Z), xShift=-10, rotations=[("xz", 90), ("yz", 90)], device=DEVICE)
    mask = maskLoader.load(visualize=True)
    

    # CYLINDER_CENTER = [N_X //5,  N_Y//2, N_Z//2]
    # CYLINDER_RADIUS = N_Z // 7
    # circle = np.sqrt(((X - CYLINDER_CENTER[0])**2 + (Z - CYLINDER_CENTER[2])**2) < CYLINDER_RADIUS).astype(bool) 
    # mask = circle & (Y >= CYLINDER_HEIGH ) & (Y <= (N_Y - CYLINDER_HEIGH))

    # mask = torch.tensor(mask).to(DEVICE)
    # L = N_Y - 2*CYLINDER_HEIGH

   
    # Visualizer and Solver
    vis = FluidVisualizer3D((N_X, N_Y, N_Z), saveOutput=True, outputFile="simulation.mp4", maxOpacity=0.1, fps=20, cmap="jet")
    lbm = LBMSolver3D(Ci, WEIGHTS, REYNOLD_NUMBER, N_DVELOCITIES, torch.ones((N_X, N_Y, N_Z)).to(DEVICE), 
                      DIRECTIONAL_VELOCITIES, RIGHT_VELOCITY, LATTICES, OPPOSITE_LATTICES, L, confined=CONFINED)
    
    
    # main loop
    for i in tqdm(range(10000)):
        f = lbm.update(mask)
        if i % N_PLOT == 0:
            density = computeDensity(f)
            macroVelocities = computeMacroVelocity(f, density, Ci)
            velocity = computeVelocityMagnitude(macroVelocities).cpu().numpy()
            curl = computeCurl(macroVelocities).cpu().numpy()
            vis.run(velocity, visualize=True)


