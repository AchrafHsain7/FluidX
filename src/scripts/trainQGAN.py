import numpy as np 
import matplotlib.pyplot as plt 
import pennylane as qml
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.QGAN import *
from utils.utils import to_bitstring

if __name__ == "__main__":

    DEVICE = "cuda"
    BATCH_SIZE = 32
    LR = 0.01
    EPOCHS = 2
    # Loading data
    latent_space = np.load("../../results/codebook_mini.npz")
    codebook = latent_space["codebook"]
    indices = latent_space["indices"]
    bins = latent_space["bins"]
    print("Codebook Shape:", codebook.shape)
    print("Bins Shape: ", bins.shape)
  
    n_qubits = 10
    n_anscilatory = 2
    n_layers = 6
    n_generators = 7
    

    device = qml.device("lightning.gpu", n_qubits)
    @qml.qnode(device, interface="torch", diff_method="parameter-shift")
    def qcircuit(noise, weights):
        weights = weights.reshape(n_layers, n_qubits)
        
        # Adding the encoding RY rotations
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)
        
        # The actual Quantum Processing Part 
        for l in range(n_layers):
            for q in range(n_qubits):
                #Ry parameter v
                qml.RY(weights[l][q], wires=q)
            for q in range(n_qubits-1):
                qml.CZ(wires=[q, q+1])

        return qml.probs(wires=list(range(n_qubits)))



    #Data Preparation
    print("PROCESSING DATASET")
    data = torch.Tensor(bins)
    data = torch.nn.functional.one_hot(data.long(), num_classes=256)
    data = torch.Tensor(data).to(DEVICE).float()
    dataset = torch.utils.data.TensorDataset(data)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    sample = next(iter(dataLoader))[0]
    #Model Creation
    print("CREATING MODEL")
    qgan = QGAN(n_generators, n_qubits, n_anscilatory, n_layers, n_qubits - n_anscilatory ,LR, BATCH_SIZE, "cuda")
    print("TRAINING MODEL")
    #Training
    qgan.train(dataLoader, EPOCHS)











