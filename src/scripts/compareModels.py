import torch 
import numpy as np 
import matplotlib.pyplot as plt
import pennylane as qml

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.LSTM import LSTM
from core.QCBMSampler import QCBMSampler
from core.QGAN import QuantumGenerator

from utils.utils import gaussian_dequantize, gaussian_quantize 


if __name__ == "__main__":

    DEVICE = "cuda"
    #Loading data
    data = np.load("../../results/codebook_mini.npz")
    codebook = torch.Tensor(data["codebook"]).to(DEVICE)
    indices = torch.Tensor(data["indices"])
    bins = data["bins"]
    bins_edges = data["edges"]
    
    #LSTM SAMPLING
    lstm = torch.load("../../results/LSTM/lstm.pt", weights_only=False)
    print("LSTM SAMPLE:", lstm.sample())
    
    # QCBM SAMPLING
    qcbm_weights = np.load("../../results/QCBM/QCBM_weights.npy")
    qcbm_sampler = QCBMSampler(qcbm_weights)
    pred = qcbm_sampler.sample()
    print("QCBM SAMPLE:", pred)

    # QGAN SAMPLING
    n_qubits = 10
    n_anscilatory = 2
    n_layers = 6
    n_generators = 7
    
    qgan = QuantumGenerator(n_generators, n_qubits, n_anscilatory, n_layers, n_qubits - n_anscilatory)
    qgan_weights = torch.load("../../results/QGAN/generator.pt", weights_only=False)
    for i, param in enumerate(qgan_weights.values()):
        qgan.quantum_parameters[i] = param
    pred = qgan.sample()
    print("QGAN SAMPLE:", pred)
    
    # PLOT ALL LATENT SPACES IN 2D AND 3D 
    # USE KNN TO COMPARE TO ACTUAL DATASET CODEBOOK 
    # USE MEAN OF CLOSEST LATENT SPACE POINTS FOR PERFORMANCE

    
