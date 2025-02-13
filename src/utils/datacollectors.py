import torch
import numpy as np



class DataCollector:
    def __init__(self, lbmSolver, visualizer, saveFrequency, savePath, configPath):
        self.dataInput = []
        self.dataOutput = []
        self.lbmSolver = lbmSolver
        self.visualizer = visualizer
        self.savePath = savePath
        self.configPath = configPath
        self.saveFrequency = saveFrequency

    def collect(self):
        ...
        #Run the data collector and save every freq the data and add to data
        #[::2, ::2, ::2]  save the "input" and "output"