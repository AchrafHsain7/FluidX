import torch
import torch.nn as nn
import pennylane as qml
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from utils.utils import gaussian_dequantize

class Discriminator(nn.Module):
    def __init__(self, input_shape=256*7, device="cuda"):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #x is shape b * 7 * 256 (one hot encoded)
        x = x.reshape(x.size(0), -1).float()
        return self.model(x)

class QuantumCircuit:
    def __init__(self, n_qubits, n_anscillatory, n_layers, device="lightning.gpu"):
        
        device = qml.device(device, n_qubits)
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
        
        self.n_qubits = n_qubits
        self.n_anscillatory = n_anscillatory
        self.n_layers = n_layers
        self.qcircuit = qcircuit

    def measure(self, noise, weights):
        probs = self.qcircuit(noise, weights)
        probs0 = probs[: 2**(self.n_qubits - self.n_anscillatory)]
        probs0 /= torch.sum(probs)  # Making the Transformation non Linear

        return probs0



class QuantumGenerator(nn.Module):
    def __init__(self, n_generators, n_qubits, n_anscillatory, n_layers, input_shape, device="cuda"):
        super().__init__()
        self.quantum_parameters = nn.ParameterList(
            [nn.Parameter(torch.rand(n_layers * n_qubits), requires_grad=True) for i in range(n_generators)]
        )
        self.n_generators = n_generators
        self.n_qubits = n_qubits
        self.n_anscillatory = n_anscillatory
        self.n_layers = n_layers
        self.input_shape = input_shape
        self.qcircuit = QuantumCircuit(n_qubits, n_anscillatory, n_layers)
        self.device = device
        data = np.load("../../results/codebook_mini.npz")
        bin_to_real = gaussian_dequantize(list(data["bins"].flatten()), data["edges"])
        self.bin_real_map = {} 
        self.edges = data["edges"]
        for b, r in zip(data["bins"].flatten(), bin_to_real):
            self.bin_real_map[int(b)] = float(r)
    
    def forward(self, x):
        patch_size = 2 ** self.input_shape #using 256 bin discretization
        outputs = torch.Tensor(x.size(0), 0).to(self.device)
        for p in self.quantum_parameters:
            patches = torch.Tensor(0, patch_size).to(self.device)
            for e in x:
                p_out = self.qcircuit.measure(e, p).float().unsqueeze(0).to(self.device)
                patches = torch.cat((patches, p_out))
                #Here a patche is batch of patches for a specific dimension in our latent space
        
            #concatenating the latent space for all batches
            outputs = torch.cat((outputs, patches), 1)

        #output shape will be b * (7*256)
        outputs = outputs.reshape((x.size(0), 7, 256))
        indices = torch.argmax(outputs, dim=2).unsqueeze(2)
        onehot_outputs = torch.nn.functional.one_hot(indices.squeeze(2), num_classes=256)  # shape: (b, 7, 256)
        return onehot_outputs
    
    def sample(self):
        noise = torch.rand(1, self.n_qubits, device=self.device) * torch.math.pi/2 
        pred_onehot = self.forward(noise)[0]
        pred_indices = torch.argmax(pred_onehot, dim=-1)
        pred = []
        for idx in pred_indices:
            idx = idx.item()
            if idx in self.bin_real_map:
                pred.append(self.bin_real_map[idx])
            else:
                pred.append((self.edges[idx] + self.edges[idx+1]) / 2)
        return torch.Tensor(pred).to(self.device)

class QGAN:
    def __init__(self, n_generators, n_qubits, n_anscillatory, n_layers, input_shape, lr=0.01, batch_size=32, device="cuda"):
        self.discriminator = Discriminator(256*7, device).to(device)
        self.generator = QuantumGenerator(n_generators, n_qubits, n_anscillatory, n_layers, input_shape, device).to(device)
        self.loss_func = nn.BCELoss()
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters() ,lr)
        self.optimizerG = torch.optim.Adam(self.generator.parameters() ,lr)
        self.real_labels = torch.full((batch_size, ), 1.0, dtype=torch.float, device=device)
        self.fake_labels = torch.full((batch_size, ), 0.0, dtype=torch.float, device=device)
        self.generation_noise = torch.rand(8, n_qubits, device=device) * math.pi/2 
        self.device = device

        self.batch_size = batch_size
        self.lr = lr
        self.n_qubits = n_qubits
        self.n_generators = n_generators
        self.n_layers = n_layers
        self.history = {"discriminator_loss":[], "generator_loss":[]}

    def train(self, dataLoader, epochs=100): 
        
        for epoch in range(epochs):
            for i, d in tqdm(enumerate(dataLoader), total=63):
                real_data = d[0].to(self.device)
                noise = torch.rand(self.batch_size, self.n_qubits, device=self.device) * math.pi/2
                fake_data = self.generator(noise)
                #training discriminator
                self.discriminator.zero_grad()
                outD_real = self.discriminator(real_data).view(-1)
                outD_fake = self.discriminator(fake_data).view(-1)
                if outD_real.size(0) == 32 and outD_fake.size(0) == 32:
                    lossD_real = self.loss_func(outD_real, self.real_labels)
                    lossD_fake = self.loss_func(outD_fake, self.fake_labels)
                else:
                    real_labels = torch.full((outD_real.size(0), ), 1.0, dtype=torch.float, device=self.device)
                    fake_labels = torch.full((outD_fake.size(0), ), 0.0, dtype=torch.float, device=self.device)
                    lossD_real = self.loss_func(outD_real, real_labels)
                    lossD_fake = self.loss_func(outD_fake, fake_labels)
               
                lossD_real.backward()
                lossD_fake.backward()
                loss_discriminator = lossD_fake + lossD_real
                self.optimizerD.step()

                #Training Generator
                self.generator.zero_grad()
                outD_fake = self.discriminator(fake_data).view(-1)
                if outD_fake.size(0) == 32:
                    lossG_fake = self.loss_func(outD_fake, self.real_labels)
                else:
                    lossG_fake = self.loss_func(outD_fake, real_labels)
                lossG_fake.backward()
                self.optimizerG.step()
                self.history["discriminator_loss"].append(loss_discriminator.item())
                self.history["generator_loss"].append(lossG_fake.item())

                
            if epoch % 1 == 0:
                print(f'Iteration: {epoch}, Discriminator Loss: {loss_discriminator:0.3f}, Generator Loss: {lossG_fake:0.3f}')

        
        plt.subplots(1, 2, 1)
        plt.plot(self.history["discriminator_loss"])
        plt.title("Discriminator Loss")
        plt.subplots(1, 2, 2)
        plt.plot(self.history["generator_loss"])
        plt.title("Generator Loss")
        plt.show()
        if input("Save model? ") == "y":
            torch.save(self.discriminator, "../../results/QGAN/discriminator.pt")
            torch.save(self.generator.state_dict(), "../../results/QGAN/generator.pt")
            plt.savefig("../../results/QGAN/losses.png")
    def sample(self, size):
        noise = torch.rand(size, self.n_qubits, device=self.device) * math.pi/2
        fake_data = self.generator(noise)
        return fake_data





