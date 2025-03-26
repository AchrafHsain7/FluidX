import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import torch
from tqdm import tqdm
import cmasher as cmr
from utils.utils import *
from core.maskReaders import *
from core.visualizers import *
from core.LBMSolvers import *


class FluidSimulator:
    def __init__(self, configFile):
        config = loadConfig(configFile)
        latticesMode = config["latticeMode"]
        #LATTICES
        LATTICES, OPPOSITE_LATTICES, Ci, WEIGHTS, DIRECTIONAL_VELOCITIES = generateLatticesConfig(latticesMode)
        WEIGHTS = torch.tensor(WEIGHTS).to(config["device"])
        Ci = torch.tensor(Ci, dtype=torch.float32).to(config["device"])
        LATTICES = torch.tensor(LATTICES).to(config["device"])
        OPPOSITE_LATTICES = torch.tensor(OPPOSITE_LATTICES).to(config["device"])
        # MASK

        if latticesMode == "D2Q9":
            X, Y = np.meshgrid(np.arange(config["Nx"]), np.arange(config["Ny"]), indexing="ij")
            maskLoader = MaskLoader2D(config)
            self.mask, L = maskLoader.getMask()
            self.vis = FluidVisualizer2D(Ci, config, cmaps=[cmr.guppy_r, cmr.iceburn, cmr.wildfire])
            self.lbm = LBMSolver2D(config, Ci, WEIGHTS,DIRECTIONAL_VELOCITIES,LATTICES, OPPOSITE_LATTICES, L)
        elif latticesMode == "D3Q19":
            X, Y, Z = np.meshgrid(np.arange(config["Nx"]), np.arange(config["Ny"]), np.arange(config["Nz"]), indexing="ij")
            maskLoader = MaskLoader3D(config)
            self.mask = maskLoader.load(visualize=False)
            # Visualizer and Solver
            self.vis = FluidVisualizer3D(config, saveOutput=True, outputFile="simulation.mp4")
            self.lbm = LBMSolver3D(config, Ci, WEIGHTS, DIRECTIONAL_VELOCITIES,LATTICES, OPPOSITE_LATTICES, config["maskCubeVolume"])
        
        self.config = config
        self.latticeMode = latticesMode
        

    def run(self):
        #main loop
        for i in tqdm(range(self.config["iterations"])):
            f = self.lbm.update(self.mask)
            if i % self.config["plotsFreq"] == 0:
                if self.latticeMode == "D2Q9":
                    self.vis.update(f)
                elif self.latticeMode == "D3Q19":
                    velocity = computeVelocityMagnitude(self.lbm.macroVelocity).cpu().numpy()
                    self.vis.run(velocity, visualize=True)

    def collectData(self, saveFreq, saveFile):
        data = []
        for i in tqdm(range(self.config["iterations"])):
            f = self.lbm.update(self.mask)
            if i % saveFreq == 0:
                density = computeDensity(f)
                macro_velocities = computeMacroVelocity(f, density, self.discretCoordinates)
                curl = computeCurl(macro_velocities)
                data.append(curl)
        
        datanp = np.array(data)
        print(datanp.shape)
        np.savez_compressed(f"{saveFile}.npz", data=datanp)
            

