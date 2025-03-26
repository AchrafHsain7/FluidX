import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.Simulator import FluidSimulator




class DataCollector:
    def __init__(self, collectionConfigs):
        self.collectionConfigs = collectionConfigs


    def collect(self):
        for c, frq, f in tqdm(self.collectionConfigs):
            print(c, frq, f)
            fs = FluidSimulator(c)
            fs.collectData(frq, f)
    
