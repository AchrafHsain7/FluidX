import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from utils.utils import computeDensity, computeEquilibrium

from abc import ABC, abstractmethod
import torch
import numpy as np




###################################################################################################################
class LBMInterface(ABC):
    def __init__(self, latticeCoordinates, weights, device="cuda") ->None:
        self.latticeCoordinates = latticeCoordinates
        self.weights = weights
        self.speedSound = 1 / np.sqrt(3.0)
        self.device = device
        self.equilibriumFluid = None
        self.macroVelocity = None
        self.density = None
        self.discreteFluid = None

    def computeDensity(self):
        self.density = torch.sum(self.discreteFluid, dim=-1)
    
    def computeMacroVelocity(self):
        macroVelocity = torch.einsum("XYD,dD->XYd", self.discreteFluid, self.latticeCoordinates)
        macroVelocity = torch.div(macroVelocity, self.density[..., torch.newaxis])
        self.macroVelocity = macroVelocity
    
    def computeEquilibrium(self):
        macroL2 = torch.norm(self.macroVelocity, p=2, dim=-1)
        macroCoordinates = torch.einsum("NXq,qd->NXd", self.macroVelocity, self.latticeCoordinates)
        fluidEquilibrium = (
            self.density[..., torch.newaxis]* self.weights[torch.newaxis, torch.newaxis, :] 
        * (   1 + macroCoordinates/(self.speedSound**2) 
            + (macroCoordinates**2)/(2*(self.speedSound**4)) 
            - macroL2[..., torch.newaxis]**2/(2* (self.speedSound**2))   ))
        
        self.equilibriumFluid = fluidEquilibrium

    

    @abstractmethod
    def update(self):
        ...





#####################################################################################################################
class LBMSolver2D(LBMInterface):
    def __init__(self, latticeCoordinates, weights, relaxation,
                  numberVelocities, initialDensity, directionalVelocities, profileVelocity,
                  verticalVelocities, latticeIndexes, oppositeIndexes, 
                    device="cuda"):
        
        super().__init__(latticeCoordinates, weights, device)
        self.relaxation = relaxation
        self.density = initialDensity
        self.numberVelocities = numberVelocities
        self.directionalVelocities = directionalVelocities
        self.profileVelocity = profileVelocity
        self.verticalVelocities = verticalVelocities
        self.latticeIndexes = latticeIndexes
        self.oppositeIndexes = oppositeIndexes
        self.discreteFluid = computeEquilibrium(self.profileVelocity, self.density, self.weights, self.latticeCoordinates)

    
    def collide(self):
        f_collision = self.discreteFluid - self.relaxation * (self.discreteFluid - self.equilibriumFluid)
        return f_collision
    
    def propagate(self, f_colision):
        # propagate velocities from discrete angles to neighbours
        f_propagated = f_colision.clone().detach().to(self.device)
        for i in range(self.numberVelocities):
            f_propagated[:, :, i] = torch.roll(
            torch.roll(f_colision[:, :, i], int(self.latticeCoordinates[0, i].tolist()), dims=0),
                int(self.latticeCoordinates[1, i].tolist()), dims=1)
            
        self.discreteFluid = f_propagated
    
    def boundaryConditions(self):
        #make the gradient 0 for the specific boundaries, default boundary condition at Right
        # Boundaries: 0:Left, 1: Top, 2: Right, 3:Bottom
        self.discreteFluid[0:, :, self.directionalVelocities[0]] =  self.equilibriumFluid[0, :, self.directionalVelocities[0]]
        return self.discreteFluid
    
    def inflow(self):
        #inflow
        self.macroVelocity[0, 1:-1, :] = self.profileVelocity[0, 1:-1, :]
        self.density[0, :] = computeDensity(self.discreteFluid[0, :, self.verticalVelocities]) + 2*computeDensity(self.discreteFluid[0, :, self.directionalVelocities[0]])
        self.density[0, :] /= (1 - self.macroVelocity[0, :, 0])
        

    def bounce(self, f_collison, mask):
        for i in range(self.numberVelocities):
            f_collison[:, :, self.latticeIndexes[i]] = (
            torch.where(mask, self.discreteFluid[:, :, self.oppositeIndexes[i]], f_collison[:, :, self.latticeIndexes[i]])
            )

        return f_collison
    

    def update(self, mask):
        #right boundary condition: flow not coming back from right boundary 
        self.discreteFluid[-1, :, self.directionalVelocities[0]] = self.discreteFluid[-2, :, self.directionalVelocities[0]]
        #compute moments and densities
        self.computeDensity()
        self.computeMacroVelocity()
        #Inflow
        self.inflow()
        #Equilibrium
        self.computeEquilibrium()
        #BC right
        self.discreteFluid[0, :, self.directionalVelocities[2]] = self.equilibriumFluid[0, :, self.directionalVelocities[2]]
        #BGK collision
        collisionFluid = self.collide()
        #Bounce back: no slip condition
        self.discreteFluid = self.bounce(collisionFluid, mask)
        #propagate
        self.propagate(self.discreteFluid)
        return self.discreteFluid

################################################################################################################







