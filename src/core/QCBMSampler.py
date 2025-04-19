import torch
import pennylane as qml
import numpy as np

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.QCBM import QCBM
from core.MMD import MMD
from utils.utils import gaussian_dequantize

class QCBMSampler:
    def __init__(self, weights, device="cuda"):
        
        self.weights = weights
        self.device = device
        n_circuits = 7
        n_qubits = 8 
        n_layers = 7       
        qml_device = qml.device("lightning.qubit", wires=n_qubits) 
        weights_total = self.weights 

        @qml.qnode(qml_device, interface="torch", diff_method="parameter-shift")
        def circuit(weights):
            #qml.BasisState(bitstring, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.probs()
        
        self.circuit = circuit
            
        data = np.load("../../results/codebook_mini.npz")
        bin_to_real = gaussian_dequantize(list(data["bins"].flatten()), data["edges"])
        self.bin_real_map = {}
        for b, r in zip(data["bins"].flatten(), bin_to_real):
            self.bin_real_map[int(b)] = float(r)
        self.edges = data["edges"]
        
    def sample(self):
        qcbm_probs = self.circuit(self.weights)
        sampled_bin = torch.multinomial(qcbm_probs, num_samples=1).squeeze(-1)
        #mapping to corresponding sampled bin index to an actual codebbok value
       
        pred = []
        for i, v in enumerate(sampled_bin):
            idx = v.item()
            if idx not in self.bin_real_map:
                pred.append((self.edges[idx] + self.edges[idx+1]) / 2)
            else:
                pred.append(self.bin_real_map[idx])

        return torch.Tensor(pred).to(self.device)

