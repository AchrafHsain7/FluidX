import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import json
import jax.numpy as jnp
from scipy.stats import norm


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

##################################################
def computeVelocityMagnitude(macroVelocities):
    return torch.linalg.norm(macroVelocities, axis=-1, ord=2)


################################################3
def computeCurl(macroVelocities):
    if len(macroVelocities.shape) == 3:
        du_dx, du_dy = torch.gradient(macroVelocities[..., 0])
        dv_dx, dv_dy = torch.gradient(macroVelocities[..., 1])
        curl = du_dy - dv_dx
    elif len(macroVelocities.shape) == 4:
        du_dx, du_dy, du_dz = torch.gradient(macroVelocities[..., 0])
        dv_dx, dv_dy, dv_dz = torch.gradient(macroVelocities[..., 1])
        dw_dx, dw_dy, dw_dz = torch.gradient(macroVelocities[..., 2])
        curl_x = dw_dy - dv_dz
        curl_y = du_dz - dw_dx
        curl_z = dv_dx - du_dy
        curl = torch.stack([curl_x, curl_y, curl_z], dim=-1)
        curl = torch.linalg.norm(curl, axis=-1)
    
    return curl

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
            "bottom": [4, 7, 8], 
            "vertical": [0, 2, 4], 
            "horizental": [0, 1, 3]
            }
    
    return lattices, oppositeLattices, latticeCoordinates, weights, directionallattices


##############################################
def showMesh3D(f):
    scene = trimesh.load(f)

    if isinstance(scene, trimesh.Scene):
        mesh = list(scene.geometry.values())[0]
    else:
        mesh = scene
    vertices = mesh.vertices
    faces = mesh.faces

    min_coords = vertices.min(axis=0)  
    max_coords = vertices.max(axis=0)  
    center = (max_coords + min_coords) / 2 
    scale = (max_coords - min_coords).max() 
    normalized_vertices = (vertices - center) / scale

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(normalized_vertices[faces], alpha=0.5, edgecolor='k'))

    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    ax.set_box_aspect([1, 1, 1])

    plt.show()

#########################################

def createCylinder(dimensions, center, radius, height=None, device="cuda"):
    if len(dimensions) == 2:
        X, Y = dimensions
        cylinder = np.sqrt((X - center[0])**2 + (Y - center[1])**2) < radius
        return torch.tensor(cylinder.astype(bool)).to(device), radius

    elif len(dimensions) == 3:
        X, Y, Z = dimensions
        cylinder = (np.sqrt((X - center[0])**2 + (Z - center[2])**2) < radius).astype(bool)
        cylinder = cylinder & (Y >= height ) & (Y <= (X.shape[1] - height))
        l = X.shape[1] - 2*height
        return torch.tensor(cylinder).to(device), l
    
#################################################33

def loadConfig(fpath):
    with open(fpath, "r") as f:
        config = json.load(f)
    return config

#######################################################
def normalized_mse(x:torch.Tensor, y:torch.Tensor):
  x_norm = (x - x.mean()) / (x.std() + 1e-8)
  y_norm = (y - y.mean()) / (y.std() + 1e-8)
  return torch.nn.functional.mse_loss(x_norm, y_norm)
###########################################################
def smap(y_pred: torch.Tensor, y_true:torch.Tensor, epsilon=1e-8):
    # symmetric mean absolute percentage error
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_pred) + torch.abs(y_true)) / 2 + epsilon
    return torch.mean(numerator / denominator)

def log_mse(y_true, y_pred, epsilon=1e-8):
    log_y_pred = torch.sign(y_pred) * torch.log(torch.abs(y_pred) + epsilon)
    log_y_true = torch.sign(y_true) * torch.log(torch.abs(y_true) + epsilon)
    
    return torch.mean((log_y_pred - log_y_true) ** 2)
###########################################################
class MMD:
    def __init__(self, bandwiths, dataSpace):
        gammas = 1 / (2 * (bandwiths**2))
        distance = jnp.abs(dataSpace[:, None] - dataSpace[None, :]) ** 2
        self.kernel = sum(jnp.exp(-gamma * distance) for gamma in gammas) * len(bandwiths)
        self.bandwiths = bandwiths

    def kernel_expval(self, px, py):
    #px is the predicted distribution
    #py is the target distribution
        return px @ self.kernel @ py

    def __call__(self, px, py):
        pxy = px - py 
        return self.kernel_expval(pxy, pxy)

###############################################################
def quantize_bin(data, min_val=None, max_val=None, nbins=255):
    data = np.array(data)
    if min_val==None or max_val == None:
        min_val, max_val = min(data), max(data)
    clipped_vals = np.clip(data, min_val, max_val)
    bins = ((clipped_vals - min_val) / (max_val - clipped_vals) * nbins).astype(np.uint8)

    return bins

def dequantize_bin(quantized_data, max_val, min_val, nbins=255):
    return (bins.astype(np.float32) / nbins) * (max_val - min_val) + min_val

##########################################################################

def gaussian_quantize(values, num_bins=256, mu=0.0, sigma=1.0, epsilon=1e-6):
    values = np.asarray(values)
    quantiles = np.linspace(epsilon, 1-epsilon, num_bins+1)
    bin_edges = norm.ppf(quantiles, loc=mu, scale=sigma)
    bins = np.digitize(values, bin_edges)
    bins = np.clip(bins, 0, num_bins - 1)
    return bins, bin_edges


def gaussian_dequantize(bins, bin_edges):
    midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return midpoints[bins]

###################################################################################
def to_bitstring(num):
  bitstring = f'{6:08b}'
  bitstring = list(bitstring)
  bitstring = [int(i) for i in bitstring]
  return bitstring

