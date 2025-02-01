import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from utils.utils import computeDensity, computeEquilibrium, computeMacroVelocity

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from mpl_toolkits import mplot3d
import cmasher as cmr
from PIL import Image, ImageOps
import numpy as np
import pyvista as pv
import time





def visualize(f, X, Y, Ci, Z=None, mode="2d", mask=None, grid=None, plotter=None):
    density = computeDensity(f)
    macro_velocities = computeMacroVelocity(f, density, Ci)

    #Computing the velocity magnitudes and Curls/Vortex Magnitudes
    velocity_magnitude = torch.linalg.norm(macro_velocities, axis=-1, ord=2)

    if mode == "2d":
        du_dx, du_dy = torch.gradient(macro_velocities[..., 0])
        dv_dx, dv_dy = torch.gradient(macro_velocities[..., 1])
        curl = du_dy - dv_dx

        velocity_magnitude = np.flip(velocity_magnitude.cpu().numpy().T, axis=0)
        curl = np.flip(curl.cpu().numpy().T, axis=0)

        #Plots
        plt.subplot(211)
        # plt.contourf(X, Y, velocity_magnitude, levels=20, cmap="jet")
        plt.imshow(velocity_magnitude, cmap="jet", aspect="equal")
        plt.colorbar().set_label("Velocity Magnitude")

        plt.subplot(212)
        plt.imshow(curl, cmap=cmr.redshift, vmin=-0.02, vmax=0.02, aspect="equal")
        plt.colorbar().set_label("Vorticity Magnitude")

        plt.tight_layout()
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
        
        


def readmask(file,Nx, Ny, device="cuda", ratio_Y=0.5, leftoffset=0):

    img = Image.open(file).convert("L")
    img = np.array(ImageOps.exif_transpose(img))
    img_ratio = img.shape[0] / img.shape[1]  # Height/Width ratio
    new_y = int(Ny * ratio_Y)
    new_x = int(new_y / img_ratio)
    img = np.array(Image.fromarray(img).resize((new_x, new_y)))

    lateral = np.sqrt(new_y**2 + new_x**2)
    lenghtCharacteristic = max(new_x, new_y, lateral)

    target_height = Ny
    target_width = Nx
    # Get the current dimensions of the image
    current_height, current_width = img.shape[:2]

    # Calculate padding for each side
    pad_top = (target_height - current_height) // 2
    pad_bottom = target_height - current_height - pad_top
    pad_left = (target_width - current_width) // 2 
    pad_right = target_width - current_width - pad_left 
    padded_img = np.pad(
        img,
        ((pad_top, pad_bottom), (pad_left - leftoffset, pad_right + leftoffset)),  # Padding for height and width
        mode="constant",
        constant_values=255  # White padding
    )
    img = padded_img < np.mean(img)
    img = img[::-1, :].T
    return torch.tensor(img.copy()).to(device), lenghtCharacteristic




class FluidVisualizer:
    def __init__(self, X, Y, Z, discretFluid, fluidCoordinates, elev=20, azim=145, resolution=1):

        self.discretFluid = discretFluid
        self.fluidCoordinates = fluidCoordinates
        self.resolution = resolution

        self.X_flat = X[::self.resolution, :, :].ravel()
        self.Y_flat = Y[::self.resolution, :, :].ravel()
        self.Z_flat = Z[::self.resolution, :, :].ravel()

        # Calculate ranges 
        self.x_range = np.max(X) - np.min(X)
        self.y_range = np.max(Y) - np.min(Y)
        self.z_range = np.max(Z) - np.min(Z)
        
        # Set up the figure and axes
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([self.x_range, self.y_range, self.z_range])
        self.ax.view_init(elev=elev, azim=azim)

        self.initialAlpha = 0.05
        # Initialize scatter plot with dummy data
        self.scatter = self.ax.scatter(
            self.X_flat, self.Y_flat, self.Z_flat,
            c=np.zeros_like(self.X_flat),
            cmap="jet",
            s=20,
            alpha=self.initialAlpha,
            rasterized=True, 
        )
        
        # Plot
        self.colorbar = plt.colorbar(self.scatter, ax=self.ax, label='Velocity Magnitude')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Real-time Fluid Flow')
        self.ax.grid(True)
        
        plt.tight_layout()



    def computeValues(self):
        #Computing the velocity magnitudes and Curls/Vortex Magnitudes
        self.density = computeDensity(self.discretFluid)
        self.macroVelocities = computeMacroVelocity(self.discretFluid, self.density, self.fluidCoordinates)
        velocityMagnitude = torch.linalg.norm(self.macroVelocities, axis=-1, ord=2)
        du_dx, du_dy, du_dz = torch.gradient(self.macroVelocities[..., 0])
        dv_dx, dv_dy, dv_dz = torch.gradient(self.macroVelocities[..., 1])
        dw_dx, dw_dy, dw_dz = torch.gradient(self.macroVelocities[..., 2])
        curl_x = dw_dy - dv_dz
        curl_y = du_dz - dw_dx
        curl_z = dv_dx - du_dy
        curl = torch.stack([curl_x, curl_y, curl_z], dim=-1)

        self.velocityMagnitude = velocityMagnitude.cpu().numpy()[::self.resolution, :, :]
        self.curl = curl.cpu().numpy()[::self.resolution, ...]
        self.curl = np.linalg.norm(self.curl, axis=-1)

        

    def update(self, discreteFluid, mask=None):
       
        self.discretFluid = discreteFluid
        self.computeValues()
        # Update mask if provided
        if mask is not None:
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            mask = mask[::self.resolution, :, :]
            mask_flat = mask.ravel()
        # Update scatter plot colors
        heatmap = self.velocityMagnitude.ravel()
        lowerCutoff = np.percentile(heatmap, 5)  
        upperCutoff = np.percentile(heatmap, 80)  # 
        # Get the least 30% and most 30% of values
        maskLower = heatmap <= lowerCutoff
        maskUpper = heatmap >= upperCutoff
        maskAlpha = maskLower  

        self.scatter.set_array(heatmap)
        self.scatter.set_alpha(np.full(heatmap.shape, self.initialAlpha))
        alpha_values = np.where(maskAlpha, 0.5, self.initialAlpha)
        self.scatter.set_alpha(alpha_values)

        # Update colorbar limits
        vmin = np.min(heatmap[heatmap > 0])
        vmax = np.max(heatmap)
        self.scatter.set_clim(vmin, vmax)
        
        # Draw and flush events
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001) 





def animate_4d_data(data, output_filename=None, fps=10, cmap='viridis', opacity=0.5):

    time_steps, x_dim, y_dim, z_dim = data.shape
    # PyVista ImageData object
    grid = pv.ImageData(dimensions=(x_dim, y_dim, z_dim))
    grid.point_data["values"] = data[0].flatten()

    # Initialize the Plotter
    plotter = pv.Plotter() 
    plotter.add_volume(grid, scalars="values", opacity=opacity, cmap=cmap)

    # Update function for each frame
    def update_frame(t):
        current_data = data[t].flatten(order="F")
        grid.point_data["values"] = current_data
        plotter.clear()
        plotter.add_volume(grid, scalars="values", opacity=opacity, cmap=cmap)
        
    # Create the animation
    if output_filename:
        plotter.open_movie(output_filename, framerate=fps, quality=10)
    for t in range(time_steps):
        update_frame(t)
        if output_filename:
            plotter.write_frame()
        else:
            plotter.render()
        

