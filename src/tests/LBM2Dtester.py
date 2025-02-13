import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.LBMSolvers import LBMSolver2D
from core.visualizers import FluidVisualizer2D
from utils.utils import *
from core.maskReaders import MaskLoader2D

import torch
import numpy as np
from tqdm import tqdm
import cmasher as cmr



if __name__ == "__main__":

    
    CONFIG = loadConfig("../../config/simulationConfigs/airplane2D.json")

    #LATTICES
    LATTICES, OPPOSITE_LATTICES, Ci, WEIGHTS, DIRECTIONAL_VELOCITIES = generateLatticesConfig("D2Q9")
    WEIGHTS = torch.tensor(WEIGHTS).to(CONFIG["device"])
    Ci = torch.tensor(Ci, dtype=torch.float32).to(CONFIG["device"])
    LATTICES = torch.tensor(LATTICES).to(CONFIG["device"])
    OPPOSITE_LATTICES = torch.tensor(OPPOSITE_LATTICES).to(CONFIG["device"])


    
    # MASK
    X, Y = np.meshgrid(np.arange(CONFIG["Nx"]), np.arange(CONFIG["Ny"]), indexing="ij")
    maskLoader = MaskLoader2D(CONFIG)
    mask, L = maskLoader.getMask()
    # mask, L = createCylinder((X, Y), (N_X//5, N_Y//2), N_Y//9)


    #SOLVER
    vis = FluidVisualizer2D(Ci, mask, CONFIG, cmaps=[cmr.guppy_r, cmr.iceburn, cmr.wildfire])
    lbm = LBMSolver2D(CONFIG, Ci, WEIGHTS,DIRECTIONAL_VELOCITIES,LATTICES, OPPOSITE_LATTICES, L)
    

    #main loop
    for i in tqdm(range(CONFIG["iterations"])):
        f = lbm.update(mask)
        if i % CONFIG["plotsFreq"] == 0:
            vis.update(f)
    
