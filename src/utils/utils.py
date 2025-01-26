import torch
import numpy as np



def computeDensity(discreteFluid) -> torch.Tensor:
    return torch.sum(discreteFluid, dim=-1)
    
def computeMacroVelocity(discreteFluid, density, latticeCoordinates) -> torch.Tensor:
    macroVelocity = torch.einsum("XYD,dD->XYd", discreteFluid, latticeCoordinates)
    macroVelocity = torch.div(macroVelocity, density[..., torch.newaxis])
    return macroVelocity

def computeEquilibrium(macroVelocity, density, weights, latticeCoordinates, speedSound=1/np.sqrt(3)):
        macroL2 = torch.norm(macroVelocity, p=2, dim=-1)
        macroCoordinates = torch.einsum("NXq,qd->NXd", macroVelocity, latticeCoordinates)
        fluidEquilibrium = (
            density[..., torch.newaxis]* weights[torch.newaxis, torch.newaxis, :] 
        * (   1 + macroCoordinates/(speedSound**2) 
            + (macroCoordinates**2)/(2*(speedSound**4)) 
            - macroL2[..., torch.newaxis]**2/(2* (speedSound**2))   ))
        return fluidEquilibrium