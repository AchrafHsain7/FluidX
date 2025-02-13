import torch
import numpy as np
from PIL import Image, ImageOps
import open3d as o3d 
from scipy.ndimage import rotate
import matplotlib.pyplot as plt



##############################################################################
class MaskLoader2D:
    def __init__(self, config):
        self.Nx = config["Nx"]
        self.Ny = config["Ny"]
        self.heightRatio = config["heightRatio"]
        self.leftOffset = config["leftOffset"]
        self.upOffset = config["upOffset"]
        self.device = config["device"]
        self.img = None
        self.maskPath = config["mask"]

    def loadMask(self):
        img = Image.open(self.maskPath).convert("L")
        img = np.array(ImageOps.exif_transpose(img))
        img_ratio = img.shape[0] / img.shape[1]  # Height/Width ratio
        self.newY = int(self.Ny * self.heightRatio)  #computing the desired height
        self.newX = int(self.newY / img_ratio)
        self.img = np.array(Image.fromarray(img).resize((self.newX, self.newY)))

    def computeCharacteristicL(self):
        lateral = np.sqrt(self.newX**2 + self.newY**2)
        characteristicL = max(self.newX, self.newY, lateral)
        return characteristicL

    def padMask(self):
        current_height, current_width = self.img.shape[:2]

        # Calculate padding for each side
        pad_top = (self.Ny - current_height) // 2
        pad_bottom = self.Ny - current_height - pad_top
        pad_left = (self.Nx - current_width) // 2 
        pad_right = self.Nx - current_width - pad_left 

        padded_img = np.pad(
            self.img,
            ((pad_top + self.upOffset, pad_bottom - self.upOffset), (pad_left + self.leftOffset, pad_right - self.leftOffset)),
            mode="constant",
            constant_values=255 
        )
        img = padded_img < np.mean(self.img)
        self.mask = img[::-1, :].T

    def getMask(self):
        self.loadMask()
        self.padMask()
        charLength = self.computeCharacteristicL()
        return torch.tensor(self.mask.copy()).to(self.device), charLength

        

##############################################################################
class MaskLoader3D:

    def __init__(self, config):
        self.filePath = config["mask"]
        self.modelDims = config["maskCubeVolume"]
        self.spaceDims = (config["Nx"], config["Ny"], config["Nz"])
        self.xShift = config["frontOffset"]
        self.rotations = config["rotations"]
        self.resolutionPrecision = config["maskResolutionPrecision"]
        self.device = config["device"]


    def loadVoxels(self):
        mesh = o3d.io.read_triangle_mesh(self.filePath)
        if mesh.is_empty(): 
            raise ValueError("The mesh loaded is empty.")
        grid_dimensions = (self.modelDims, self.modelDims, self.modelDims)
        voxel_array = np.zeros(grid_dimensions, dtype=np.float32)
        voxel_size = 100

        while True:
            try: 
                voxel_array = np.zeros(grid_dimensions, dtype=np.float32)
                voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
                voxels = voxel_grid.get_voxels()
                indices = np.array([voxel.grid_index for voxel in voxels])
                for idx in indices:
                    voxel_array[tuple(idx)] = True
                voxel_size *= self.resolutionPrecision

            except:
                voxel_array = np.zeros(grid_dimensions, dtype=np.float32)
                voxel_size /= self.resolutionPrecision
                voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
                voxels = voxel_grid.get_voxels()
                indices = np.array([voxel.grid_index for voxel in voxels])
                for idx in indices:
                    voxel_array[tuple(idx)] = True
                break

        self.voxels = voxel_array.copy()
        return voxel_array
    
    def rotate(self):
        for r, ang in self.rotations:
            if r == "xz":
                self.voxels = rotate(self.voxels, angle=ang, axes=(0, 2), reshape=False, order=1)
            elif r == "xy":
                self.voxels = rotate(self.voxels, angle=ang, axes=(0, 1), reshape=False, order=1)
            elif r == "yz":
                self.voxels = rotate(self.voxels, angle=ang, axes=(1, 2), reshape=False, order=1)
            else:
                print("Oops")

    def paddings(self):
        pd_x = self.spaceDims[0] - self.modelDims
        pd_y = self.spaceDims[1] - self.modelDims
        pd_z = self.spaceDims[2] - self.modelDims
        self.voxels = np.pad(self.voxels, pad_width=((0, pd_x), (0, pd_y), (0, pd_z)), mode="constant", constant_values=False)

    def center(self):
        indices = np.stack(np.where(self.voxels == True), axis=-1) 
        med_bound = np.median(indices, axis=0)

        x_shift = (self.spaceDims[0] // 2) - med_bound[0] + 1
        y_shift = (self.spaceDims[1] // 2) - med_bound[1] + 1
        z_shift = (self.spaceDims[2] // 2) - med_bound[2] + 1

        self.voxels = np.roll(self.voxels, shift=(x_shift, y_shift, z_shift), axis=(0, 1, 2))
        self.voxels = np.roll(self.voxels, shift=self.xShift, axis=0) 

    
    def plot(self):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(111, projection="3d")
        x, y, z = np.where(self.voxels)
        ax.scatter(x, y, z, c="black", marker="s")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim(0, self.modelDims)
        ax.set_ylim(0, self.modelDims)
        ax.set_zlim(0, self.modelDims)

        plt.show()


    def load(self, visualize=False):

        self.loadVoxels()
        self.rotate()
        if visualize:
            self.plot()
        self.paddings()
        self.center()
        
        
        return torch.tensor(self.voxels.astype(bool)).to(self.device)
        


    

          





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