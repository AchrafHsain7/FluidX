import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.QCBM import QCBM



class QCBMTrainer:
    def __init__(self, weights, bitstrings, prior, qml_device, circuit, mmd, n_iterations=100, lr=0.1, device="cuda"):
        # Convert weights to PyTorch tensor with requires_grad=True and move to CUDA
        # Use torch.nn.Parameter to ensure it's properly tracked
        self.weights = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32, device=device))
        self.optimizer = torch.optim.Adam([self.weights], lr=lr)
        
        # Set up the quantum circuit with the torch interface
        self.circ = circuit
        self.prior = torch.tensor(prior, dtype=torch.float32, device=device)
        self.history = []
        self.divs = []
        self.n_iterations = n_iterations
        batch_bits = []
        
        for j in range(len(bitstrings)):
            bitstring = bitstrings[j]
            bitstring = torch.tensor([int(i) for i in list(bitstring)], device=device)
            batch_bits.append(bitstring)
            
        self.batch_bits = torch.stack(batch_bits) if batch_bits else torch.tensor([], device=device)
        
        # Create QCBM object for loss calculations
        self.qcbm = QCBM(self.circ, mmd, self.prior)
        
    def train(self):
        for i in range(self.n_iterations):
            self.optimizer.zero_grad()
            bitstring = self.batch_bits[i % len(self.batch_bits)]
           # Calculate loss and get circuit output
            loss, qcbm_probs = self.qcbm.mmd_loss(self.weights, bitstring)
           #Computing KL divergence metric
            with torch.no_grad():
                eps = 1e-10  
                safe_probs = qcbm_probs + eps
                safe_prior = self.prior + eps
                kl_div = -torch.sum(self.prior * torch.log(safe_probs / safe_prior))
            # Backward pass
            loss.backward()
            # Update parameters
            self.optimizer.step()
            # Save metrics
            if i % 10 == 0:
                print(f"Step: {i} Loss: {loss.item():.4f} KL-div: {kl_div.item():.4f}")
            self.history.append(loss.item())
            self.divs.append(kl_div.item())

        
    def draw_results(self, save_path):
        
        if not os.path.exist(save_path):
            os.mkdir(save_path)
        # Plot training history and save to file instead of showing
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(self.history)
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("MMD Loss")
        ax[1].plot(self.divs, color="green")
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("KL Divergence")
        plt.tight_layout()
        history_plot_path = os.path.join(save_path, "training_history.png")
        plt.savefig(history_plot_path)
        plt.close(fig)
        
        # Comparing the true prior to the learned distribution
        with torch.no_grad():
            qcbm_probs = self.circ(self.weights, self.batch_bits[0]).detach().cpu().numpy()
            prior_cpu = self.prior.cpu().numpy()
        
        plt.figure(figsize=(12, 5))
        plt.bar(
            np.arange(256),
            prior_cpu,
            width=2.0,
            label=r"$\pi(x)$",
            alpha=0.4,
            color="tab:blue",
        )
        plt.bar(
            np.arange(256),
            qcbm_probs,
            width=2.0,
            label=r"$p_\theta(x)$",
            alpha=0.9,
            color="tab:green",
        )
        plt.xlabel("Samples")
        plt.ylabel("Prob. Distribution")
        plt.legend(loc="upper right")
        plt.subplots_adjust(bottom=0.3)
        plt.tight_layout()
        dist_plot_path = os.path.join(save_path, "distribution_comparison.png")
        plt.savefig(dist_plot_path)
        plt.close()


