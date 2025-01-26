import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
import cmasher as cmr
import sys
import jax.numpy as jnp
from tqdm import tqdm


# DEVICE PARAMETERS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: ", DEVICE)
PRECISION = torch.float64
torch.set_default_dtype(PRECISION)

# Discretizing space, Time, and velocities
N_X = 400
N_Y = 150
N_ITERATIONS = 15_000
N_PLOT = 50
DELTA_T = 1 #seconds
DELTA_X = 1 #1 pixel delta X

# D2Q9 set of locations
r"""
6 2 5
3 0 1
7 4 8
"""
N_DVELOCITIES = 9 #number of discrete velocities
WEIGHTS = torch.tensor([4, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25]).to(DEVICE) / 9
LATTICES = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
OPPOSITE_LATTICES = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6])
#0 Left, 1: Top, 2: Right, 3: Bottom
DIRECTIONAL_VELOCITIES = torch.tensor([
    [3, 6, 7],#left
    [2, 5, 6],#top
    [1, 5, 8],#right
    [4, 7, 8]#bottom
]).to(DEVICE)
VERTICAL_VELOCITIES = torch.tensor([0, 2, 4]).to(DEVICE)
HORIZENTAL_VELOCITIES = torch.tensor([0, 1, 3]).to(DEVICE)



#SIMULATION PARAMETERS

RIGHT_VELOCITY = 0.05 #mach
SOUND_SPEED = DELTA_X / (DELTA_T*np.sqrt(3))
# RELAXATION_TIME = 0.1  # TODO: Check how it is actuallu computed
CYLINDER_RADIUS = N_Y // 9
REYNOLD_NUMBER = 275
kinematic_viscosity = (RIGHT_VELOCITY * CYLINDER_RADIUS) / REYNOLD_NUMBER
RELAXATION_TIME = 1.0 / (3 * kinematic_viscosity + 0.5)  # TODO: Check how it is actuallu computed


MASK = "../models/airplane.jpg"






################################################################

Ci = torch.tensor([
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1]
], dtype=PRECISION).to(DEVICE)


def compute_density(f):
    #Input: Discrete velocities
    #Output: Density for each lattice

    return torch.sum(f, axis=-1)

def compute_macro(f, density):
    #Input: Discrete Velocities and density
    #Output: Macro velocities (x and y axis)
    macro_vel = torch.einsum("YXD,dD->YXd", f, Ci)
    macro_vel = torch.div(macro_vel, density[..., torch.newaxis])
    return macro_vel

def compute_equilibrium(macro_vel, density):
    #Input: Macro velocities and Density
    #Output: Equilibrium discrete velocities
    
    macro_l2 = torch.norm(macro_vel, p=2, dim=-1)
    macro_Ci = torch.einsum("NXq,qd->NXd", macro_vel, Ci)
    feq = (
        density[..., torch.newaxis]* WEIGHTS[torch.newaxis, torch.newaxis, :] 
     * (   1 + macro_Ci/(SOUND_SPEED**2) 
        + (macro_Ci**2)/(2*(SOUND_SPEED**4)) 
        - macro_l2[..., torch.newaxis]**2/(2* (SOUND_SPEED**2))   )
    )
    return feq

def collide(f, feq):
    f_collision = f - RELAXATION_TIME * (f - feq)
    return f_collision


def propagate(f_colision):
    # propagate velocities from discrete angles to neighbours
    f_propagated = torch.tensor(f_colision).to(DEVICE)
    for i in range(N_DVELOCITIES):
        f_propagated[:, :, i] = torch.roll(
        torch.roll(f_colision[:, :, i], int(Ci[0, i].tolist()), dims=0),
            int(Ci[1, i].tolist()), dims=1)
        
    return f_propagated
    
def boundary_condition(f, feq, boundary=0):
    #make the gradient 0 for the specific boundaries, default boundary condition at Right
    # Boundaries: 0:Left, 1: Top, 2: Right, 3:Bottom
    f[0:, :, DIRECTIONAL_VELOCITIES[boundary]] =  feq[0, :, DIRECTIONAL_VELOCITIES[boundary]]
    return f

    
def dirichlet_inflow(macro_vel, profile_velocity, f, density):
    #inflow
    macro_vel[0, 1:-1, :] = profile_velocity[0, 1:-1, :]
    density[0, :] = compute_density(f[0, :, VERTICAL_VELOCITIES]) + 2*compute_density(f[0, :, DIRECTIONAL_VELOCITIES[0]])
    density[0, :] /= (1 - macro_vel[0, :, 0])
    return macro_vel, density



def bounce_back(f_collison, f, mask):
    for i in range(N_DVELOCITIES):
        f_collison[:, :, LATTICES[i]] = torch.where(mask, f[:, :, OPPOSITE_LATTICES[i]], f_collison[:, :, LATTICES[i]])

    return f_collison


def timstep(f, profile_vel, mask):
     #right boundary condition: flow not coming back from right boundary 
     f[-1, :, DIRECTIONAL_VELOCITIES[0]] = f[-2, :, DIRECTIONAL_VELOCITIES[0]]


     #compute moments and densities
     density = compute_density(f)
     macro_vel = compute_macro(f, density)
     
     #Inflow
     macro_vel, density = dirichlet_inflow(macro_vel, profile_vel, f, density)

    #  u =  macro_vel[:, :, 0].cpu().numpy()
    #  v = macro_vel[:, :, 1].cpu().numpy()
    #  print(X.shape, Y.shape, u.shape, v.shape)


    #  plt.quiver(X, Y, u,v , color="red",  scale_units="xy")
    # #  plt.streamplot(Y, X, u, v,color=u**2 + v**2,  cmap="viridis")
    #  plt.show()

    #  print("Step3: =================================")
    #  print(macro_vel)
    #  print(density)

    

     #Equilibrium
     feq = compute_equilibrium(macro_vel, density)
    #  print("Step4: ===================")
    #  print(feq)



     #BC right
     f[0, :, DIRECTIONAL_VELOCITIES[2]] = feq[0, :, DIRECTIONAL_VELOCITIES[2]]
    #  print("Step 5: =======================")
    #  print(f[0, :, DIRECTIONAL_VELOCITIES[2]])


     #BGK collision
     f_final = collide(f, feq)
    #  print("Step 6: ========================")
    #  print(f_final[mask])
    #  sys.exit()


     #Bounce back: no slip condition
     f_final = bounce_back(f_final, f, mask)
    #  print("Step 7: =================")
    #  print(f_final)



     #propagate
     f_final = propagate(f_final)

     return f_final


    

def readmask(file,Nx, Ny, ratio_Y=0.5):

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
        ((pad_top, pad_bottom), (pad_left, pad_right)),  # Padding for height and width
        mode="constant",
        constant_values=255  # White padding
    )
    img = padded_img < np.mean(img)
    img = img[::-1, :].T
    return torch.tensor(img.copy()).to(DEVICE)


def visualize(f, X, Y, mask):
    density = compute_density(f)
    macro_velocities = compute_macro(f, density)

    #Computing the velocity magnitudes and Curls/Vortex Magnitudes
    velocity_magnitude = torch.linalg.norm(macro_velocities, axis=-1, ord=2)
    du_dx, du_dy = torch.gradient(macro_velocities[..., 0])
    dv_dx, dv_dy = torch.gradient(macro_velocities[..., 1])
    curl = du_dy - dv_dx

    velocity_magnitude = velocity_magnitude.cpu()
    curl = curl.cpu()

    # u =  macro_velocities[:, :, 0].cpu().numpy()
    # v = macro_velocities[:, :, 1].cpu().numpy()
    # print(X.shape, Y.shape, u.shape, v.shape)


    # plt.quiver(X, Y, u,v , color="blue",  scale_units="xy")
    # #  plt.streamplot(Y, X, u, v,color=u**2 + v**2,  cmap="viridis")
    # plt.show()

    

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
    








if __name__ == "__main__":

    velcity_profile = torch.zeros((N_X, N_Y, 2)).to(DEVICE)
    velcity_profile[:, :, 0] = RIGHT_VELOCITY

    F = compute_equilibrium(velcity_profile, torch.ones(N_X, N_Y).to(DEVICE))

    #CYLINDER
    X, Y = np.meshgrid(np.arange(N_X), np.arange(N_Y), indexing="ij")
    CYLINDER_CENTER = [N_X // 5, N_Y // 2]
    CYLINDER_RADIUS = N_Y // 9
    mask = np.sqrt(
        (X - CYLINDER_CENTER[0])**2 + (Y - CYLINDER_CENTER[1])**2
        ) < CYLINDER_RADIUS
    mask = torch.tensor(mask).to(DEVICE)

    # mask = readmask(MASK, N_X, N_Y, ratio_Y=0.5)


    for i in tqdm(range(N_ITERATIONS)):
        F = timstep(F, velcity_profile, mask)
        if i % N_PLOT == 0:
            visualize(F, X, Y, mask)




