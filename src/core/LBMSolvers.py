import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from utils.utils import computeDensity, computeEquilibrium, computeMacroVelocity

from abc import ABC, abstractmethod
import torch
import numpy as np




###################################################################################################################
class LBMInterface(ABC):
    def __init__(self, latticeCoordinates, weights, reynoldNumber,
                  numberVelocities, initialDensity, directionalVelocities, rightVelocity,
                  latticeIndexes, oppositeIndexes, characteristicLength,
                    device="cuda", confined=False) ->None:
        
        self.latticeCoordinates = latticeCoordinates
        self.weights = weights
        self.speedSound = 1 / np.sqrt(3.0)
        self.device = device
        self.confined = confined
        self.equilibriumFluid = None
        self.macroVelocity = None
        self.kinematicViscosity = (rightVelocity * characteristicLength) / reynoldNumber
        self.relaxation = 1.0 / (3 * self.kinematicViscosity + 0.5)
        self.density = initialDensity
        self.numberVelocities = numberVelocities
        self.directionalVelocities = directionalVelocities
        self.latticeIndexes = latticeIndexes
        self.oppositeIndexes = oppositeIndexes

    def computeDensity(self):
        self.density = torch.sum(self.discreteFluid, dim=-1)
    
    def computeMacroVelocity(self):
        macroVelocity = computeMacroVelocity(self.discreteFluid, self.density, self.latticeCoordinates)
        self.macroVelocity = macroVelocity
    
    def computeEquilibrium(self):
        fluidEquilibrium = computeEquilibrium(self.macroVelocity, self.density, self.weights, self.latticeCoordinates, self.speedSound)
        self.equilibriumFluid = fluidEquilibrium

    

    @abstractmethod
    def update(self):
        ...





#####################################################################################################################
class LBMSolver2D(LBMInterface):
    def __init__(self, latticeCoordinates, weights, reynoldNumber,
                  numberVelocities, initialDensity, directionalVelocities, rightVelocity,
                latticeIndexes, oppositeIndexes, characteristicLength,
                    device="cuda", confined=False):
        super().__init__(latticeCoordinates, weights, reynoldNumber,
                  numberVelocities, initialDensity, directionalVelocities, rightVelocity,
                  latticeIndexes, oppositeIndexes, characteristicLength,
                    device, confined)
        
        self.profileVelocity =  torch.zeros((initialDensity.shape[0], initialDensity.shape[1],  2)).to(device)
        self.profileVelocity[:, :, 0] = rightVelocity
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
        self.discreteFluid[0:, :, self.directionalVelocities["left"]] =  self.equilibriumFluid[0, :, self.directionalVelocities["left"]]
        return self.discreteFluid
    
    def inflow(self):
        #inflow
        self.macroVelocity[0, 1:-1, :] = self.profileVelocity[0, 1:-1, :]
        self.density[0, :] = computeDensity(self.discreteFluid[0, :, self.directionalVelocities["vertical"]]) + 2*computeDensity(self.discreteFluid[0, :, self.directionalVelocities["left"]])
        self.density[0, :] /= (1 - self.macroVelocity[0, :, 0])
        

    def bounce(self, f_collison, mask):
        for i in range(self.numberVelocities):
            f_collison[:, :, self.latticeIndexes[i]] = (
            torch.where(mask, self.discreteFluid[:, :, self.oppositeIndexes[i]], f_collison[:, :, self.latticeIndexes[i]])
            )

        return f_collison
    

    def update(self, mask):
        #right boundary condition: flow not coming back from right boundary 
        self.discreteFluid[-1, :, self.directionalVelocities["left"]] = self.discreteFluid[-2, :, self.directionalVelocities["left"]]
        if self.confined:
            self.discreteFluid[:, -1, self.directionalVelocities["bottom"]] = self.discreteFluid[:, -2, self.directionalVelocities["bottom"]]
            self.discreteFluid[: , 0, self.directionalVelocities["top"]] = self.discreteFluid[:, 1, self.directionalVelocities["top"]]
        #compute moments and densities
        self.computeDensity()
        self.computeMacroVelocity()
        #Inflow
        self.inflow()
        #Equilibrium
        self.computeEquilibrium()
        #BC right
        self.discreteFluid[0, :, self.directionalVelocities["right"]] = self.equilibriumFluid[0, :, self.directionalVelocities["right"]]
        #BGK collision
        collisionFluid = self.collide()
        #Bounce back: no slip condition
        self.discreteFluid = self.bounce(collisionFluid, mask)
        #propagate
        self.propagate(self.discreteFluid)
        return self.discreteFluid

################################################################################################################


class LBMSolver3D(LBMInterface):
    def __init__(self, latticeCoordinates, weights, reynoldNumber,
                  numberVelocities, initialDensity, directionalVelocities, rightVelocity,
                  latticeIndexes, oppositeIndexes, characteristicLength,
                    device="cuda", confined=False):
        super().__init__(latticeCoordinates, weights, reynoldNumber,
                  numberVelocities, initialDensity, directionalVelocities, rightVelocity,
                  latticeIndexes, oppositeIndexes, characteristicLength,
                    device, confined)
        
        self.profileVelocity =  torch.zeros((initialDensity.shape[0], initialDensity.shape[1], initialDensity.shape[2], 3)).to(device)
        self.profileVelocity[:, :, :, 0] = rightVelocity
        self.discreteFluid = computeEquilibrium(self.profileVelocity, self.density, self.weights, self.latticeCoordinates)
        
        
    def propagate(self, f_colision):
        f_propagated = f_colision.clone().detach().to(self.device)
        for i in range(self.numberVelocities):
            f_propagated[:, :, :, i] = torch.roll(
                                        torch.roll(
                                            torch.roll(f_colision[:, :, :,  i], int(self.latticeCoordinates[0, i].tolist()), dims=0),
                                            int(self.latticeCoordinates[1, i].tolist()), dims=1),
                                        int(self.latticeCoordinates[2, i].tolist()), dims=2) 
            
        self.discreteFluid = f_propagated

    def boundaryConditions(self):
        #make the gradient 0 for the specific boundaries, default boundary condition at Right
        # Boundaries: 0:Left, 1: Top, 2: Right, 3:Bottom
        self.discreteFluid[0:, :, :, self.directionalVelocities["front"]] =  self.equilibriumFluid[0, :, :, self.directionalVelocities["front"]]
        return self.discreteFluid
    
    def collide(self):
        f_collision = self.discreteFluid - self.relaxation * (self.discreteFluid - self.equilibriumFluid)
        return f_collision
    
    def inflow(self):
        self.macroVelocity[0, 1:-1, 1:-1, :] = self.profileVelocity[0, 1:-1, 1:-1, :]
        self.density[0, :, :] = (
            # Base density from non-inlet velocities
            computeDensity(self.discreteFluid[0, :, :, self.directionalVelocities["noninlet"]]) +
            # Double contribution from inlet velocities 
            2*computeDensity(self.discreteFluid[0, :, :, self.directionalVelocities["inlet"]])
        )

                            
        self.density[0, :, :] /= (1 - self.macroVelocity[0, :, :, 0])
    
    def bounce(self, f_collison, mask):
        for i in range(self.numberVelocities):
            f_collison[:, :, :, self.latticeIndexes[i]] = (
            torch.where(mask, self.discreteFluid[:, :, :, self.oppositeIndexes[i]], f_collison[:, :, :, self.latticeIndexes[i]])
            )

        return f_collison
    
    def update(self, mask):
        #right boundary condition: flow not coming back from front boundary 
        self.discreteFluid[-1, :, :, self.directionalVelocities["back"]] = self.discreteFluid[-2, :, :, self.directionalVelocities["back"]]
        #compute moments and densities
        self.computeDensity()
        self.computeMacroVelocity()
        #Inflow
        self.inflow()
        #Equilibrium
        self.computeEquilibrium()
        #BC right
        self.discreteFluid[0, :, :, self.directionalVelocities["front"]] = self.equilibriumFluid[0, :, :, self.directionalVelocities["front"]]
        #BGK collision
        collisionFluid = self.collide()
        #Bounce back: no slip condition
        self.discreteFluid = self.bounce(collisionFluid, mask)
        #propagate
        self.propagate(self.discreteFluid)
        return self.discreteFluid




