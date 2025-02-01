import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh



##############################################
def computeDensity(discreteFluid) -> torch.Tensor:
    return torch.sum(discreteFluid, dim=-1)


############################################## 
def computeMacroVelocity(discreteFluid, density, latticeCoordinates) -> torch.Tensor:
     if len(discreteFluid.shape) == 3:
        macroVelocity = torch.einsum("XYD,dD->XYd", discreteFluid, latticeCoordinates)
     else:
         macroVelocity = torch.einsum("XYZD,dD->XYZd", discreteFluid, latticeCoordinates)
     macroVelocity = torch.div(macroVelocity, density[..., torch.newaxis])
     return macroVelocity

##############################################
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


##############################################
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
            # inletVelocities = [1, 7, 9, 11, 13]
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


##############################################
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




