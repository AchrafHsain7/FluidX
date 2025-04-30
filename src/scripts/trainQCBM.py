import torch
import pennylane as qml
import numpy as np
import time
from functools import partial
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.MMD import MMD
from core.QCBM import QCBM
from core.QCBMTrainer import QCBMTrainer

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    # Create a results directory
    results_dir = "../../results"
    os.makedirs(results_dir, exist_ok=True)
        # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Loading data
        latent_space = np.load("../../results/codebook_mini.npz")
        codebook = latent_space["codebook"]
        indices = latent_space["indices"]
        bins = latent_space["bins"]
        print("Codebook Shape:", codebook.shape)
        print("Bins Shape: ", bins.shape)
        
        # Creating the Circuit
        n_circuits = 7
        n_qubits = 8 
        n_layers = 7
       
        qml_device = qml.device("lightning.qubit", wires=n_qubits) 
        wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        weights = np.random.random(size=wshape)
        
        @qml.qnode(qml_device, interface="torch", diff_method="parameter-shift")
        def circuit(weights, bitstring):
            #qml.BasisState(bitstring, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.probs()
        
        print("DRAWING CIRCUIT")
        qml.draw_mpl(circuit, level="device",max_length=7)(weights, None)
        plt.show()
        sys.exit()

        bandwidths = torch.tensor([0.25, 0.5, 1.0], device=device)
        space = torch.arange(256, device=device)
        mmd = MMD(bandwidths, space)
        
        # Creating the prior distribution
        priors = []  
        bitstrings = []
        for i in range(n_circuits):
            quantized_val, count = np.unique(bins[:, i], return_counts=True)
            prior = np.zeros(256)
            prior[quantized_val] = count / sum(count)
            priors.append(prior)
            bitstrings.append([])
            for bv in quantized_val:
                bitstrings[i].append(format(bv, f'0{8}b')) 
        bitstrings.insert(0, ["0" * 8])
        
        n_iterations = 100 
        model_weights = []
        
        for i in range(n_circuits):
            print(f"\n=== Training circuit {i+1}/{n_circuits} ===")
            trainer = QCBMTrainer(weights, bitstrings[i], priors[i], qml_device, circuit, mmd, n_iterations=n_iterations, lr=0.1)
            # Train the model
            trainer.train()
            circuit_results_dir = os.path.join(results_dir, f"QCBM/QCBM_plots/circuit_{i}")
            trainer.draw_results(save_path=circuit_results_dir)
            model_weights.append(trainer.weights.detach().cpu().numpy())
        
        # Save all model weights
        weights_path = os.path.join(results_dir, "QCBM/QCBM_weights.npy")
        np.save(weights_path, np.array(model_weights))
        print(f"Model weights saved to {weights_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
