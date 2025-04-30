import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from utils.utils import quantize_bin, dequantize_bin, gaussian_quantize, gaussian_dequantize

if __name__ == "__main__":
    latent_space = np.load("../../results/codebook_mini.npz")
    codebook = latent_space["codebook"]
    indices = latent_space["indices"]
    shape = codebook.shape
    print(codebook.shape)
    print(indices.shape)
    
    # Codeword frequency distribution
    indices, counts = np.unique(indices, return_counts=True)
    codewords_distribution = np.zeros(128)
    codewords_distribution[indices] = counts
    #plt.bar(np.arange(128), codewords_distribution)
    #plt.show()

    #Codewords numbers
    codewords_values = []
    for cw in codebook:
        codewords_values.extend(list(cw))
    print("CODEWORDS VALUES: " + str(len(codewords_values)))
    print("UNIQUE VALUES: " + str(len(np.unique(codewords_values))))
    plt.hist(codewords_values, bins=30)
    plt.show()

    # Strategy: Train 24 models each with 8qbits precision, each model will have to learn the distributions at element i from 24 from all codewords_values
    # This will be an autoregressive teaching forcing variation over QCBM to learn the conditional distribution. Beware inference time, error accumulation and data shift
    bins, edges = gaussian_quantize(codewords_values)
    plt.hist(bins, bins=30)
    plt.show()
    original = gaussian_dequantize(bins, edges)
    plt.hist(original, bins=30)
    plt.show()
    bins = bins.reshape(shape)
    # from bins to bitsttrings for QCBM
    tsne = TSNE(n_components=2, perplexity=20).fit_transform(codebook)
    print(tsne.shape)
    plt.scatter(tsne[:, 0], tsne[:, 1])
    plt.show()

    





    

