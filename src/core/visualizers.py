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
    def __init__(self, discretCoordinates, mask):
        self.discretCoordinates = discretCoordinates
        self.mask = mask

    def computations(self, discretFluid):
        density = computeDensity(discretFluid)
        macro_velocities = computeMacroVelocity(discretFluid, density, self.discretCoordinates)

        velocity_magnitude = computeVelocityMagnitude(macro_velocities)
        curl = computeCurl(macro_velocities, mode="2d")

        self.velocityMagnitude = np.flip(velocity_magnitude.cpu().numpy().T, axis=0)
        self.curl = np.flip(curl.cpu().numpy().T, axis=0)

    def plot(self):
        plt.subplot(211)
        plt.imshow(self.velocityMagnitude, cmap="jet", aspect="equal")
        plt.colorbar().set_label("Velocity Magnitude")
        plt.subplot(212)
        plt.imshow(self.curl, cmap=cmr.redshift, vmin=-0.02, vmax=0.02, aspect="equal")
        plt.colorbar().set_label("Vorticity Magnitude")

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        plt.clf()

    def update(self, discretFluid):
        self.computations(discretFluid)
        self.plot()



##########################################33
class FluidVisualizer3D:
    def __init__(self, dimensions, saveOutput=False, outputFile=None, fps=10, cmap="jet", maxOpacity=0.1):
        self.data = None
        self.dimensions = dimensions
        self.timeSteps = 0
        self.saveOutput = saveOutput
        self.outputFile = outputFile
        self.fps = fps
        self.cmap = cmap
        opacitiesL = np.linspace(0, maxOpacity, 50).tolist()
        opacitiesR = np.linspace(0, maxOpacity, 50).tolist()
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

    
    def run(self, data, visualize=False):
        self.update(data)
        if self.saveOutput:
            self.plotter.write_frame()
        if visualize:
            self.plotter.render()


######################################        




        

