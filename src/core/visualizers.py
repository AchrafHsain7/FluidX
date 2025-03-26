import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from utils.utils import *

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cmasher as cmr
import numpy as np
import pyvista as pv




        

#############################################3
class FluidVisualizer2D:
    def __init__(self, discretCoordinates, config, cmaps=[cmr.infinity, cmr.iceburn, cmr.fusion]):
        self.discretCoordinates = discretCoordinates
        self.plotNums = config["plotNums"]
        self.clims = config["colorLims"]
        self.cmaps = cmaps
        plt.style.use('dark_background')


    def computations(self, discretFluid):
        density = computeDensity(discretFluid)
        macro_velocities = computeMacroVelocity(discretFluid, density, self.discretCoordinates)

        velocity_magnitude = computeVelocityMagnitude(macro_velocities)
        curl = computeCurl(macro_velocities)

        self.velocityMagnitude = np.flip(velocity_magnitude.cpu().numpy().T, axis=0)
        self.curl = np.flip(curl.cpu().numpy().T, axis=0)
        self.density = np.flip(density.cpu().numpy().T, axis=0)

    def plot(self):
        if self.plotNums == 3:
            plt.subplot(311)
            plt.imshow(self.velocityMagnitude, cmap=self.cmaps[0], aspect="equal")
            plt.colorbar().set_label("Velocity Magnitude")
            plt.subplot(312)
            plt.imshow(self.curl, cmap=self.cmaps[1], vmin=self.clims[0], vmax=self.clims[1], aspect="equal")
            plt.colorbar().set_label("Vorticity Magnitude")
            plt.subplot(313)
            plt.imshow(self.density, cmap=self.cmaps[2], aspect="equal", vmin=0.97, vmax=1.03)
            plt.colorbar().set_label("Density")
        elif self.plotNums == 2:
            plt.subplot(211)
            plt.imshow(self.velocityMagnitude, cmap=self.cmaps[0], aspect="equal")
            plt.colorbar().set_label("Velocity Magnitude")
            plt.subplot(212)
            plt.imshow(self.curl, cmap=self.cmaps[1], vmin=-0.02, vmax=0.02, aspect="equal")
            plt.colorbar().set_label("Vorticity Magnitude")
        else:
            plt.imshow(self.velocityMagnitude, cmap=self.cmaps[0], aspect="equal")
            plt.colorbar().set_label("Velocity Magnitude")

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        plt.clf()

    def update(self, discretFluid):
        self.computations(discretFluid)
        self.plot()



##########################################33
class FluidVisualizer3D:
    def __init__(self, config, saveOutput=False, outputFile=None):
        self.data = None
        self.dimensions = (config["Nx"], config["Ny"], config["Nz"])
        self.timeSteps = 0
        self.saveOutput = saveOutput
        self.outputFile = outputFile
        self.fps = config["visualizationFPS"]
        self.cmap = config["visualizationColorMap"]
        opacitiesL = np.linspace(0, config["visualizationMaxOpacity"], 50).tolist()
        opacitiesR = np.linspace(0, config["visualizationMaxOpacity"], 50).tolist()
        self.opacities = opacitiesL[::-1] + opacitiesR
        self.animation = []

        self.grid = pv.ImageData(dimensions=self.dimensions)
        self.plotter = pv.Plotter()
        if self.saveOutput:
            self.plotter.open_movie(self.outputFile, framerate=self.fps, quality=10)

    def update(self, data):
        self.grid.point_data["values"] = data.flatten(order="F")
        self.plotter.clear()
        self.plotter.add_volume(self.grid, scalars="values", opacity=self.opacities, cmap=self.cmap)

    
    def run(self, data, dataPlot="velocity", visualize=False):
        self.update(data)
        if self.saveOutput:
            self.plotter.write_frame()
        if visualize:
            self.plotter.render()


######################################        




        

