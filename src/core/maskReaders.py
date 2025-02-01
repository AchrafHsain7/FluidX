import torch
import numpy as np
from PIL import Image, ImageOps



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