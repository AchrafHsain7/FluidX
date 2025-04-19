import torch
import torch.nn as nn
import numpy as np 


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.VQ_VAE import VQ_VAE
from utils.utils import  gaussian_quantize,gaussian_dequantize 

def collectCodebook(model, dataLoader):
    with torch.no_grad():
        codebook = []
        indices_list = []
        for img in dataloader:
            _, _, indices = model(img[0])
            codebook.append(model.embeddings(indices))
            indices_list.append(indices)
        return torch.cat(codebook, dim=0), torch.cat(indices_list, dim=0)


if __name__ == "__main__":
   #Loading model
    model = torch.load("../../results/modelVQVAE_mini.pt", weights_only=False).cuda()
    data = np.load("../../data/superResolution/cylinderData.npz")["data"]
    data = torch.Tensor(data).cuda()
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    codebook, indices = collectCodebook(model, dataloader)
    shape = codebook.shape
    codebook_flat = codebook.cpu().numpy().flatten()
    bins, edges = gaussian_quantize(codebook_flat)
    bins = bins.reshape(shape)
    print(bins.shape)
    np.savez("../../results/codebook_mini", codebook=codebook.cpu().numpy(), indices=indices.cpu().numpy(), bins=bins, edges=edges)
