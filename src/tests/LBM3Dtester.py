import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.Simulator import FluidSimulator




if __name__ == "__main__":
    simulator = FluidSimulator("../../config/simulationConfigs/airplane3D.json")
    simulator.run()
