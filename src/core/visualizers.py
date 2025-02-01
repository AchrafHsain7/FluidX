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
import open3d as o3d 
from scipy.ndimage import zoom, rotate
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





def animate_4d_data(data, output_filename=None, fps=10, cmap='jet', opacity=0.7):

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
        #dynamic opacities
        opacities = np.linspace(0, opacity, 127).tolist()
        opacities = opacities[::-1] + opacities

        plotter.add_volume(grid, scalars="values", opacity=opacities, cmap=cmap)
        
    # Create the animation
    if output_filename:
        plotter.open_movie(output_filename, framerate=fps, quality=10)
    for t in range(time_steps):
        update_frame(t)
        if output_filename:
            plotter.write_frame()
        else:
            plotter.render()





######################################        

def voxel3d(fpath, model_dims=100, space_dims=[200, 200, 200], x_shiftf=0, rotations=["xz"], resolutionPrecision=0.5):
    mesh = o3d.io.read_triangle_mesh(fpath)

    # Ensure the mesh is not empty
    if mesh.is_empty():
        raise ValueError("The mesh is empty. Please check the input file.")

    model_dims = (model_dims, model_dims, model_dims)
    grid_dimensions = model_dims

    # Initialize a 3D NumPy array to represent the voxel grid
    voxel_array = np.zeros(grid_dimensions, dtype=np.float32)

    
    # Populate the NumPy array with the voxel data
    voxel_size = 100 # Adjust based on desired resolution
    # print("Loading 3D ....")
    while True:
        try: 
            voxel_array = np.zeros(grid_dimensions, dtype=np.float32)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
            # Get voxel indices
            voxels = voxel_grid.get_voxels()
            indices = np.array([voxel.grid_index for voxel in voxels])

            for idx in indices:
                voxel_array[tuple(idx)] = True

            voxel_size *= resolutionPrecision
            # print("Current Voxel Size: ", voxel_size)

        except:
            voxel_array = np.zeros(grid_dimensions, dtype=np.float32)
            voxel_size /= resolutionPrecision
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
            # Get voxel indices
            voxels = voxel_grid.get_voxels()
            indices = np.array([voxel.grid_index for voxel in voxels])

            for idx in indices:
                voxel_array[tuple(idx)] = True
            break
    
    # print("Voxel Size: ", voxel_size)


    
    # voxel_array = zoom(voxel_array, scale_factor, order=1)

    

    #rotating
    voxel_array = rotate(voxel_array, angle=90, axes=(0, 2), reshape=False, order=1)
    voxel_array = rotate(voxel_array, angle=90, axes=(1, 2), reshape=False, order=1)

    #padd into space dimensions
    pd_x = space_dims[0] - model_dims[0]
    pd_y = space_dims[1] - model_dims[1]
    pd_z = space_dims[2] - model_dims[2]
    voxel_array = np.pad(voxel_array, pad_width=((0, pd_x), (0, pd_y), (0, pd_z)), mode="constant", constant_values=False)


    #centering the airplane
    indices = np.stack(np.where(voxel_array == True), axis=-1)
    # print(indices)
    # print(indices.shape) 
    med_bound = np.median(indices, axis=0)
    # print("Medians: ", med_bound)
    #y center
    y_center_indices = med_bound[1]
    y_center_axis = space_dims[1] // 2
    y_shift = y_center_axis - y_center_indices + 1

    z_center_indices = med_bound[2]
    z_center_axis = space_dims[2] // 2
    z_shift = z_center_axis - z_center_indices + 1

    x_center_indices = med_bound[0]
    x_center_axis = space_dims[0] // 2
    x_shift = x_center_axis - x_center_indices + 1
    
    # print("Axis center is: ", x_center_axis)
    # print("Model center is: ", x_center_indices)
    # print("Shiftin Axis by", x_shift)

    voxel_array = np.roll(voxel_array, shift=(x_shift, y_shift, z_shift), axis=(0, 1, 2))
    voxel_array = np.roll(voxel_array, shift=x_shiftf, axis=0)   


    

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Get voxel positions where voxel_array is True
    x, y, z = np.where(voxel_array)

    # x, y, z = rotateAxes(x, y, z, rotations)

    ax.scatter(x, y, z, c="black", marker="s")

    # Labels and viewing angle
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


    ax.set_xlim(0, voxel_array.shape[0])
    ax.set_ylim(0, voxel_array.shape[1])
    ax.set_zlim(0, voxel_array.shape[2])


    # plt.show()
    return voxel_array.astype(bool)


        

