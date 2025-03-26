import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from utils.utils import show3D, voxel3d

if __name__ == "__main__":
    FILE = "../../data/models/airplane.glb"
    # show3D(FILE)
    vx = voxel3d(FILE)
    