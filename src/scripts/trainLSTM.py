import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from core.LSTM import LSTM


if __name__ == "__main__":
    
    DEVICE = "cuda"
    HIDDEN_LSTM = 256
    EPOCHS = 100

    data = np.load("../../results/codebook_mini.npz")
    codebook = data["codebook"]
    indices = data["indices"]
    bins = data["bins"]
    
    #Data shape is 1999,7 of discritized bined data 
    # Training Procedure:
    # Each layer get 1 float number and output 1 float number 
    codebook = torch.Tensor(codebook.reshape((-1, 7, 1))).to(DEVICE)
    dataset = torch.utils.data.TensorDataset(codebook)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model = LSTM(HIDDEN_LSTM, 7).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss(reduction="mean")
    history = []

    for e in range(EPOCHS):
        for d in dataloader:
            x = d[0]
            predicted = model(x)
            loss = loss_func(x, predicted)
            history.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        if e % 10 == 0:
            print(f"EPOCH {e+(e==0)} LOSS {loss.item()} ")
    
    plt.plot(history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    if input("Save ?") == "y":
        torch.save(model, "../../results/LSTM/lstm.pt")
        plt.plot(history)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")    
        plt.savefig("../../results/LSTM/loss.png")
    
    print(model.sample())

