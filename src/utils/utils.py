import torch
import torch.nn.functional as F
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import zoom, rotate
import sys


def computeDensity(discreteFluid) -> torch.Tensor:
    return torch.sum(discreteFluid, dim=-1)
    
def computeMacroVelocity(discreteFluid, density, latticeCoordinates) -> torch.Tensor:
     if len(discreteFluid.shape) == 3:
        macroVelocity = torch.einsum("XYD,dD->XYd", discreteFluid, latticeCoordinates)
     else:
         macroVelocity = torch.einsum("XYZD,dD->XYZd", discreteFluid, latticeCoordinates)
     macroVelocity = torch.div(macroVelocity, density[..., torch.newaxis])
     return macroVelocity

def computeEquilibrium(macroVelocity, density, weights, latticeCoordinates, speedSound=1/np.sqrt(3)):
        macroL2 = torch.norm(macroVelocity, p=2, dim=-1)
        if len(macroVelocity.shape) == 3:
            macroCoordinates = torch.einsum("NXq,qd->NXd", macroVelocity, latticeCoordinates)
            fluidEquilibrium = (
                density[..., torch.newaxis]* weights[torch.newaxis, torch.newaxis, :] 
            * (   1 + macroCoordinates/(speedSound**2) 
                + (macroCoordinates**2)/(2*(speedSound**4)) 
                - macroL2[..., torch.newaxis]**2/(2* (speedSound**2))   ))
        else:
            macroCoordinates = torch.einsum("XYZq,qd->XYZd", macroVelocity, latticeCoordinates)
            fluidEquilibrium = (
                density[..., torch.newaxis]* weights[torch.newaxis, torch.newaxis, :] 
            * (   1 + macroCoordinates/(speedSound**2) 
                + (macroCoordinates**2)/(2*(speedSound**4)) 
                - macroL2[..., torch.newaxis]**2/(2* (speedSound**2))   ))
            
        return fluidEquilibrium


def generateLatticesConfig(model: str):
    if model.upper() == "D3Q19":
        latticeCoordinates = [  [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
                                [0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1, -1, 1, -1],
                                [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, -1, 1, 1, -1, -1, 1]
                ]
        weights = [1/3] + 6*[1/18] + 12*[1/36]
        lattices = np.arange(19)
        oppositeLattices = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]
        leftlattices, rightlattices, toplattices, bottomlattices, frontlattices, backlattices = [], [], [], [], [], []
    
        for i in lattices:
            if latticeCoordinates[1][i] == -1:
                leftlattices.append(i)
            if latticeCoordinates[1][i] == 1:
                rightlattices.append(i)

            if latticeCoordinates[2][i] == -1:
                bottomlattices.append(i)
            if latticeCoordinates[2][i] == 1:
                toplattices.append(i)
            
            if latticeCoordinates[0][i] == 1:
                frontlattices.append(i)
            if latticeCoordinates[0][i] == -1:
                backlattices.append(i)

            inletVelocities = [2, 8, 10, 12, 14]
            nonInletVelocities = [0, 3, 4, 5, 6, 15, 16, 17, 18]

        directionallattices = {"left": leftlattices, "right":rightlattices, "top":toplattices, "bottom":bottomlattices, "front": frontlattices, "back":backlattices, 
                               "inlet": inletVelocities, "noninlet":nonInletVelocities }

    elif model.upper() == "D2Q9":
        weights = list(np.array([4, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25], dtype=np.float32)  / 9)
        lattices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        oppositeLattices = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        latticeCoordinates = [
            [0, 1, 0, -1, 0, 1, -1, -1, 1],
            [0, 0, 1, 0, -1, 1, 1, -1, -1]
        ]
        directionallattices = {
            "left": [3, 6, 7],
            "top": [2, 5, 6],
            "right": [1, 5, 8],
            "bottom": [4, 7, 8]
            }
    
    return lattices, oppositeLattices, latticeCoordinates, weights, directionallattices



def show3D(f):
    scene = trimesh.load(f)

    # If the file contains a single mesh, extract it
    if isinstance(scene, trimesh.Scene):
        # Access the first geometry in the scene
        mesh = list(scene.geometry.values())[0]
    else:
        # Directly assign the mesh if it's not a Scene
        mesh = scene

    # Get the vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    # Normalize the vertices
    min_coords = vertices.min(axis=0)  # Minimum along each axis
    max_coords = vertices.max(axis=0)  # Maximum along each axis
    center = (max_coords + min_coords) / 2  # Center of the bounding box
    scale = (max_coords - min_coords).max()  # Maximum range (for uniform scaling)

    # Translate vertices to the origin and scale them to fit within [-0.5, 0.5]^3
    normalized_vertices = (vertices - center) / scale

    # Plot using Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add the normalized mesh to the plot
    ax.add_collection3d(Poly3DCollection(normalized_vertices[faces], alpha=0.5, edgecolor='k'))

    # Set axis limits for the normalized plot
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)

    # Set the box aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    plt.show()



import open3d as o3d


def rotateAxes(x, y, z, rotations):
    for rotation in rotations:
        if rotation == "xy":
            x, y, z =  y, x, z
        elif rotation == "xz":
            x, y, z = z, y, x
        
        elif rotation == "yz":
            x, y, z = x, z, y
        
        else:
            print("Oops...")

    return x, y, z
            


def voxel3d(fpath, model_dims=[100, 100, 100], space_dims=[200, 200, 200], x_shiftf=0, rotations=["xz"], scale_factor = 1):
    mesh = o3d.io.read_triangle_mesh(fpath)

    # Ensure the mesh is not empty
    if mesh.is_empty():
        raise ValueError("The mesh is empty. Please check the input file.")

    
    grid_dimensions = model_dims

    # Initialize a 3D NumPy array to represent the voxel grid
    voxel_array = np.zeros(grid_dimensions, dtype=np.float32)

    
    # Populate the NumPy array with the voxel data
    voxel_size = 100 # Adjust based on desired resolution
    resolutionPrecision = 0.3
    print("Loading 3D ....")
    while True:
        try: 
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
    
    print("Voxel Size: ", voxel_size)


    
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
    print(indices)
    print(indices.shape) 
    med_bound = np.median(indices, axis=0)
    print("Medians: ", med_bound)
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


    plt.show()
    return voxel_array


