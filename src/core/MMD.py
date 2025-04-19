import torch 

class MMD:
    def __init__(self, bandwidths, dataSpace, device="cuda"):
        gammas = 1 / (2 * (bandwidths**2))
        # Move tensors to CUDA
        self.dataSpace = dataSpace.to(device)
        distance = torch.abs(self.dataSpace[:, None] - self.dataSpace[None, :]) ** 2
        self.kernel = sum(torch.exp(-gamma * distance) for gamma in gammas).double() * len(bandwidths)
        self.bandwidths = bandwidths.to(device)
        
    def kernel_expval(self, px, py):
        # px is the predicted distribution
        # py is the target distribution
        return px @ self.kernel @ py
        
    def __call__(self, px, py):
        pxy = px - py 
        return self.kernel_expval(pxy, pxy)


