import torch 
import numpy as np 
import matplotlib.pyplot as plt
import pennylane as qml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, pairwise_distances
from sklearn.utils import resample
import seaborn as sns
sns.set()
sns.set_style("whitegrid")


import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.LSTM import LSTM
from core.QCBMSampler import QCBMSampler
from core.QGAN import QuantumGenerator

from utils.utils import gaussian_dequantize, gaussian_quantize 


if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLE_SIZE = 1999
    #Loading data
    data = np.load("../../results/codebook_mini.npz")
    codebook = torch.Tensor(data["codebook"]).to(DEVICE)
    indices = torch.Tensor(data["indices"])
    bins = data["bins"]
    bins_edges = data["edges"]
    

    with torch.no_grad():
        #LSTM SAMPLING
        print("LSTM SAMPLING ...")
        lstm_samples = []
        lstm = torch.load("../../results/LSTM/lstm.pt", weights_only=False)
        #print("LSTM SAMPLE:", lstm.sample())
        for i in range(SAMPLE_SIZE):
            lstm_samples.append(lstm.sample().cpu().numpy())
        
        print("QCBM SAMPLING ...")
        # QCBM SAMPLING
        qcbm_samples = []
        qcbm_weights = np.load("../../results/QCBM/QCBM_weights.npy")
        qcbm_sampler = QCBMSampler(qcbm_weights)
        for i in range(SAMPLE_SIZE):
            qcbm_samples.append(qcbm_sampler.sample().cpu().numpy())
        #pred = qcbm_sampler.sample()
        #print("QCBM SAMPLE:", pred)

        # QGAN SAMPLING
        print("QGAN SAMPLING ...")
        qgan_samples = []
        n_qubits = 10
        n_anscilatory = 2
        n_layers = 6
        n_generators = 7
        qgan = QuantumGenerator(n_generators, n_qubits, n_anscilatory, n_layers, n_qubits - n_anscilatory)
        qgan_weights = torch.load("../../results/QGAN/generator.pt", weights_only=False)
        for i, param in enumerate(qgan_weights.values()):
            qgan.quantum_parameters[i] = param
        for i in range(SAMPLE_SIZE):
            qgan_samples.append(qgan.sample().cpu().numpy())
        #pred = qgan.sample()
        #print("QGAN SAMPLE:", pred)
    

    # Data Processing
    lstm_samples = np.array(lstm_samples)
    qcbm_samples = np.array(qcbm_samples)
    qgan_samples = np.array(qgan_samples)
    # PLOT ALL LATENT SPACES IN 2D AND 3D
    all_data = np.concatenate([lstm_samples, qcbm_samples, qgan_samples], axis=0)
    print(all_data.shape)
    
    # TSNE VISUALIZATION
    data_2d = TSNE(n_components=2, perplexity=100).fit_transform(all_data)
    plt.figure(figsize=(10, 8))
    plt.scatter(data_2d[:SAMPLE_SIZE, 0], data_2d[:SAMPLE_SIZE, 1], c="b", label="LSTM")
    plt.scatter(data_2d[SAMPLE_SIZE:2*SAMPLE_SIZE, 0], data_2d[SAMPLE_SIZE:2*SAMPLE_SIZE, 1], c="r", label="QCBM")
    plt.scatter(data_2d[2*SAMPLE_SIZE:, 0], data_2d[2*SAMPLE_SIZE:, 1], c="g", label="QGAN")
    plt.title("t-SNE Visualization of Model Samples")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.savefig(f"../../results/FINAL/t_SNE_{SAMPLE_SIZE}.png")
    plt.show()

    lstm_tsne = TSNE(n_components=2, perplexity=100).fit_transform(lstm_samples)
    plt.figure(figsize=(10, 8))
    plt.scatter(data_2d[:SAMPLE_SIZE, 0], data_2d[:SAMPLE_SIZE, 1], c="b", label="LSTM")
    plt.title("t-SNE Visualization of Model Samples")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.savefig(f"../../results/FINAL/t_SNE_LSTM_{SAMPLE_SIZE}.png")
    plt.show()




    #PCA VISUALIZATION
    data_2d = PCA(n_components=2).fit_transform(all_data)
    plt.figure(figsize=(10, 8))
    plt.scatter(data_2d[SAMPLE_SIZE:2*SAMPLE_SIZE, 0], data_2d[SAMPLE_SIZE:2*SAMPLE_SIZE, 1], c="r", label="QCBM", alpha=0.7, s=20)
    plt.scatter(data_2d[2*SAMPLE_SIZE:, 0], data_2d[2*SAMPLE_SIZE:, 1], c="g", label="QGAN", alpha=0.7, s=20)
    plt.scatter(data_2d[:SAMPLE_SIZE, 0], data_2d[:SAMPLE_SIZE, 1], c="b", label="LSTM", alpha=0.7, s=20)
    plt.title("PCA Visualization of Model Samples")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(f"../../results/FINAL/pca_{SAMPLE_SIZE}.png")
    plt.legend()
    plt.show()


    # USE MEAN OF CLOSEST LATENT SPACE POINTS FOR PERFORMANCE
    codebook = codebook.cpu().numpy()
    lstm_min_avg_distance = np.mean(np.min(pairwise_distances(codebook, lstm_samples), axis=1))
    qcbm_min_avg_distances = np.mean(np.min(pairwise_distances(codebook, qcbm_samples), axis=1))
    qgan_min_avg_distances = np.mean(np.min(pairwise_distances(codebook, qgan_samples), axis=1))

    models = ["LSTM", "QCBM", "QGAN"]
    distances = [lstm_min_avg_distance, qcbm_min_avg_distances, qgan_min_avg_distances]
    plt.bar(models, distances)
    plt.title('Average Minimum Distance Between Original and Generated')
    plt.ylabel('Average Minimum Distance')
    plt.savefig(f"../../results/FINAL/AMD_{SAMPLE_SIZE}.png")
    plt.show()
    

    #Nearest Neighbor Analysis
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(np.concatenate([lstm_samples, qcbm_samples, qgan_samples], axis=0))
    distances, indices = nn.kneighbors(codebook)
    closest_models_counts = {"LSTM": 0, "QCBM":0, "QGAN":0}

    for idx in indices[:, 0]:
        if idx < SAMPLE_SIZE:
            closest_models_counts["LSTM"] += 1 
        elif idx < 2 * SAMPLE_SIZE:
            closest_models_counts["QCBM"] += 1 
        elif idx < 3 * SAMPLE_SIZE:
            closest_models_counts["QGAN"] += 1 
        else:
            print("ERROR")

    plt.bar(closest_models_counts.keys(), closest_models_counts.values(), color="orange")
    plt.title('Models Providing Closest Match to Original Data')
    plt.ylabel('Count of Closest Matches')
    plt.savefig("../../results/FINAL/closest_model.png")
    plt.show()

    # Distance Distribution for each model 
    nn_lstm = NearestNeighbors(n_neighbors=2)
    nn_lstm.fit(lstm_samples)
    lstm_distances, _ = nn_lstm.kneighbors(codebook)

    nn_qcbm = NearestNeighbors(n_neighbors=2)
    nn_qcbm.fit(qcbm_samples)
    qcbm_distances, _ = nn_qcbm.kneighbors(codebook)

    nn_qgan = NearestNeighbors(n_neighbors=2)
    nn_qgan.fit(qgan_samples)
    qgan_distances, _ = nn_qgan.kneighbors(codebook)

    plt.figure(figsize=(12, 6))
    sns.histplot(lstm_distances[:, 0], alpha=0.5, label='LSTM', bins=70, color="b", kde=True)
    sns.histplot(qcbm_distances[:, 0], alpha=0.5, label='QCBM', bins=70, color="r", kde=True)
    sns.histplot(qgan_distances[:, 0], alpha=0.5, label='QGAN', bins=70, color="g", kde=True)
    plt.legend()
    plt.title('Distribution of Distances from Original Data to Each Model')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.savefig("../../results/FINAL/distances_distribution.png")
    plt.show()






