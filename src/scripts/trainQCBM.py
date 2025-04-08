import jax
import jax.numpy as jnp
import pennylane as qml
import optax
import  numpy as np
import matplotlib.pyplot as plt
import time 

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from utils.utils import MMD
from core.QCBM import QCBM


if __name__ == "__main__":
    
    #Loading data
    latent_space = np.load("../../results/codebook_mini.npy")
    print(latent_space.shape)

    # Creating the Cucruit 
    n_qubits = 7 # since 2**7 is 128 and we only want to select from codebook and not sample from the continious latent space
    device = qml.device("lightning.gpu", wires=n_qubits)
    n_layers = 6
    wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    weights = np.random.random(size=wshape)


    @qml.qnode(device)
    def circuit(weights):
        qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
        return qml.probs()

    #Creating the Just in time compiled circuit
    jit_circuit = jax.jit(circuit)
    
    # Loss setting
    bandwiths = jnp.array([0.25, 0.5, 1])
    space = np.arange(128)
    mmd = MMD(bandwiths, space)
    codewords, count = np.unique(latent_space, return_counts=True)
    
    # Creating the prior distribution
    prior = np.zeros(128)
    prior[codewords] = count / sum(count)
    qcbm = QCBM(jit_circuit, mmd, prior)

    #Optimizer settings
    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(weights)
    circ = qml.QNode(circuit, device)
    
    # Training Step
    @jax.jit
    def update_step(params, opt_state):
        (loss, qcbm_probs), grads = jax.value_and_grad(qcbm.mmd_loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        kl_div = -jnp.sum(qcbm.prior * jnp.nan_to_num(jnp.log(qcbm_probs / qcbm.prior)))
        return params, opt_state, loss, kl_div

    # Start the Actual Training
    history = []
    divs = []
    n_iterations = 100

    for i in range(n_iterations):
        weights, opt_state, loss, kl_div = update_step(weights, opt_state)

        if i%10 == 0:
            print(f"Step: {i} Loss: {loss:.4f} KL-div: {kl_div:.4f}")
        history.append(loss)
        divs.append(kl_div)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(history)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("MMD Loss")

    ax[1].plot(divs, color="green")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("KL Divergence")
    plt.show()

    qcbm_probs = np.array(qcbm.circuit(weights))

    plt.figure(figsize=(12, 5))

    plt.bar(
        np.arange(128),
        prior,
        width=2.0,
        label=r"$\pi(x)$",
        alpha=0.4,
        color="tab:blue",
    )
    plt.bar(
        np.arange(128),
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
    plt.show()


