import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.Simulator import FluidSimulator



if __name__ == "__main__":
    fname = "airplane2D"
    if len(sys.argv) >= 2:
        fname = sys.argv[-1]
    fs = FluidSimulator(f"../../config/trainingConfigs/{fname}.json")
    fs.run()
    
    