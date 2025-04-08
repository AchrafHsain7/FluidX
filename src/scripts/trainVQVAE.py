import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
import sys, os
import cmasher as cmr
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.VQ_VAE import VQ_VAE, VQ_VAE_CNN
from utils.utils import normalized_mse




# TO DO: 
# ADD Physic informed loss
# Add perceptual loss
# Add frequency domain loss
# Larger Network + Residual Connections ?



if __name__ == "__main__":
    # LOADING THE DATA
    data = np.load("../../data/superResolution/cylinderData.npz")
    Xtrain = torch.Tensor(data["data"]).cuda()
    #print(Xtrain.shape) # 1999, 256, 64

    trainDataset = torch.utils.data.TensorDataset(Xtrain)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)
    
    EPOCHS = 50
    PLOTS = {"loss":[], "qloss":[], "closs":[]} #quantized and construction loss
    OUTPUTS = []

    MINI = True

    if MINI:
        model = VQ_VAE(128, 24).cuda()
    else:
        model = VQ_VAE(128, 32).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    lossFunc = normalized_mse
    lossScaling = 1
    for e in tqdm(range(EPOCHS)):
        for img in trainLoader:
            img = img[0]
            generatedImg, qloss, _ = model(img)
            loss = lossFunc(img, generatedImg) / lossScaling 
            PLOTS["closs"].append(loss.item())
            loss = loss + qloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            PLOTS["loss"].append(loss.item())
            PLOTS["qloss"].append(qloss.item())
            if np.random.rand() < 0.25:
                OUTPUTS.append((img.cpu().detach(), generatedImg.cpu().detach()))
    

    #PLotting Results
    fig, ax = plt.subplots()
    ax.plot(PLOTS["loss"], label="Total Loss")
    ax.plot(PLOTS["qloss"], label="Quantized")
    ax.plot(PLOTS["closs"], label="Reconstruction")
    plt.legend()
    plt.show()
    del PLOTS   
    # Showing Sample Images
    for k in range(0, EPOCHS+1, 5):
        plt.figure(figsize=(15, 5))
        plt.title(f"Epoch {k}")
        for i in range(9):
            j = np.random.randint(0, len(OUTPUTS[k][0]))
            plt.subplot(2, 9, i+1)
            plt.imshow(OUTPUTS[k][0][j], cmap=cmr.iceburn, vmin=-0.02, vmax=0.02)
            plt.subplot(2, 9, (i+1)+9)
            plt.imshow(OUTPUTS[k][1][j], cmap=cmr.iceburn)

        plt.tight_layout()
        plt.show()
    choice = input("Save? ")
    if choice.lower() == "y":
        if MINI:
            torch.save(model, "../../results/modelVQVAE_mini.pt")
            fig.savefig("../../results/TrainingLoss_mini.png", dpi=300)
        else:
            torch.save(model, "../../results/modelVQVAE.pt")
            fig.savefig("../../results/TrainingLoss.png", dpi=300)
        



