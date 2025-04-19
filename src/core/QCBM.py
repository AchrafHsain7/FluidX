class QCBM:
    def __init__(self, circuit, mmd, prior, device="cuda"):
        self.circuit = circuit
        self.mmd = mmd
        self.prior = prior.to(device)
        
    def mmd_loss(self, params, bitstring):
        # Get probabilities from quantum circuit - these automatically have gradients in PyTorch interface
        px = self.circuit(params, bitstring)
        
        # Calculate MMD loss
        loss = self.mmd(px, self.prior)
        return loss, px


