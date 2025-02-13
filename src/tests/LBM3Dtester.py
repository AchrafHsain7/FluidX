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


    CONFIG = loadConfig("../../config/simulationConfigs/airplane3D.json")
    LATTICES, OPPOSITE_LATTICES, Ci, WEIGHTS, DIRECTIONAL_VELOCITIES = generateLatticesConfig("D3Q19")

    WEIGHTS = torch.tensor(WEIGHTS).to(CONFIG["device"])
    Ci = torch.tensor(Ci, dtype=torch.float32).to(CONFIG["device"])
    LATTICES = torch.tensor(LATTICES).to(CONFIG["device"])
    OPPOSITE_LATTICES = torch.tensor(OPPOSITE_LATTICES).to(CONFIG["device"])


    X, Y, Z = np.meshgrid(np.arange(CONFIG["Nx"]), np.arange(CONFIG["Ny"]), np.arange(CONFIG["Nz"]), indexing="ij")
    maskLoader = MaskLoader3D(CONFIG)
    mask = maskLoader.load(visualize=True)
    

   
    # Visualizer and Solver
    vis = FluidVisualizer3D(CONFIG, saveOutput=True, outputFile="simulation.mp4")
    lbm = LBMSolver3D(CONFIG, Ci, WEIGHTS, DIRECTIONAL_VELOCITIES,LATTICES, OPPOSITE_LATTICES, CONFIG["maskCubeVolume"])
    
    
    # main loop
    for i in tqdm(range(10000)):
        f = lbm.update(mask)
        if i % CONFIG["plotsFreq"] == 0:
            density = computeDensity(f)
            macroVelocities = computeMacroVelocity(f, density, Ci)
            velocity = computeVelocityMagnitude(macroVelocities).cpu().numpy()
            curl = computeCurl(macroVelocities).cpu().numpy()
            vis.run(velocity, visualize=True)


