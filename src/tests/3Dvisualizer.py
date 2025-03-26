import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from utils.utils import showMesh3D

if __name__ == "__main__":
    FILE = "../../data/models/airplane.glb"
    # show3D(FILE)
    vx = showMesh3D(FILE)
    