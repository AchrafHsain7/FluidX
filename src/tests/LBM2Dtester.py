import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.LBMSolvers import LBMSolver2D
from core.visualizers import FluidVisualizer2D
from utils.utils import generateLatticesConfig
from core.maskReaders import MaskLoader2D

import torch
import numpy as np
from tqdm import tqdm



if __name__ == "__main__":

    #GENERAL
    N_X, N_Y = 500, 300
    DEVICE = "cuda"
    PRECISION = torch.float32
    N_DVELOCITIES = 9 
    
    #LATTICES
    LATTICES, OPPOSITE_LATTICES, Ci, WEIGHTS, DIRECTIONAL_VELOCITIES = generateLatticesConfig("D2Q9")
    WEIGHTS = torch.tensor(WEIGHTS).to(DEVICE)
    Ci = torch.tensor(Ci, dtype=PRECISION).to(DEVICE)
    LATTICES = torch.tensor(LATTICES).to(DEVICE)
    OPPOSITE_LATTICES = torch.tensor(OPPOSITE_LATTICES).to(DEVICE)


    #SIMULATION PARAMETERS
    N_ITERATIONS = 10_000
    N_PLOT = 150
    RIGHT_VELOCITY = 0.02 #mach
    CYLINDER_RADIUS = N_Y // 9
    REYNOLD_NUMBER = 5000
    MASK = "../../data/models/airplane.jpg"
    HEIGH_RATIO = 0.5
    CONFINED_MODE = True
    
    # MASK
    X, Y = np.meshgrid(np.arange(N_X), np.arange(N_Y), indexing="ij")
    maskLoader = MaskLoader2D(N_X, N_Y, HEIGH_RATIO, leftOffset=-100, device=DEVICE)
    mask, L = maskLoader.getMask(MASK)


    #SOLVER
    vis = FluidVisualizer2D(Ci, mask)
    lbm = LBMSolver2D(Ci, WEIGHTS, REYNOLD_NUMBER, N_DVELOCITIES, torch.ones((N_X, N_Y)).to(DEVICE), 
                      DIRECTIONAL_VELOCITIES, RIGHT_VELOCITY, LATTICES, 
                      OPPOSITE_LATTICES, L,  confined=CONFINED_MODE)
    

    #main loop
    for i in tqdm(range(N_ITERATIONS)):
        f = lbm.update(mask)
        if i % N_PLOT == 0:
            vis.update(f)
    
