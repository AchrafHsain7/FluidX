import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from utils.utils import computeDensity, computeEquilibrium, computeMacroVelocity

import torch
import matplotlib.pyplot as plt
import cmasher as cmr
from PIL import Image, ImageOps
import numpy as np



def visualize(f, X, Y, Ci):
    density = computeDensity(f)
    macro_velocities = computeMacroVelocity(f, density, Ci)

    #Computing the velocity magnitudes and Curls/Vortex Magnitudes
    velocity_magnitude = torch.linalg.norm(macro_velocities, axis=-1, ord=2)
    du_dx, du_dy = torch.gradient(macro_velocities[..., 0])
    dv_dx, dv_dy = torch.gradient(macro_velocities[..., 1])
    curl = du_dy - dv_dx

    velocity_magnitude = velocity_magnitude.cpu()
    curl = curl.cpu()

    #Plots
    plt.subplot(211)
    plt.contourf(X, Y, velocity_magnitude, levels=50, cmap="jet")
    plt.colorbar().set_label("Velocity Magnitude")
    plt.subplot(212)
    plt.contourf(X, Y, curl, levels=50, cmap=cmr.redshift, vmin=-0.02, vmax=0.02)
    plt.colorbar().set_label("Vorticity Magnitude")
    
    # plt.tight_layout()
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
    return torch.tensor(img.copy()).to(device)