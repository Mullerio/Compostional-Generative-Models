import torch
print("torch module path:", torch.__file__)
print("torch module dir:", dir(torch))
from abc import ABC, abstractmethod
from scipy.stats import norm
import numpy as np


"""Abstract classes for probability, including schedules for gaussian paths"""

def rect_gauss_overlap_mass(center: torch.Tensor, std: float, rectangle: list[tuple[float, float]]) -> float:
    """
    Estimate the probability mass of a multivariate Gaussian within a rectangle. We assume isotropic Gaussians. 

    Args:
        center: torch.Tensor of shape [D]  mean of the Gaussian
        std: float standard deviation (assumed same in all dimensions, since isotropic)
        rectangle: list of (low, high) tuples, one per dimension

    Returns:
        float: estimated probability mass inside the rectangle
    """
    center_np = center.detach().cpu().numpy()
    probs = []
    for d in range(len(center_np)):
        low, high = rectangle[d]
        p = norm.cdf(high, loc=center_np[d], scale=std) - norm.cdf(low, loc=center_np[d], scale=std)
        probs.append(p)
    return float(np.prod(probs))

class LogDensity(ABC):
    """
    Abstract class for probability densities where we know log density and its score. 
    """
    @property
    @abstractmethod
    def log_density(self, x : torch.Tensor) -> torch.Tensor:
        """
        Computes log(p(x))
        Args:
            x (torch.Tensor): some point of shape [batch_size, dim]

        Returns:
            torch.Tensor: log(p(x)) [batch_size, 1]
        """
        
        pass

    def score(self, x : torch.Tensor) -> torch.Tensor:
        """
        Returns the Gradient log density(x) i.e. the score
        Args:
            x: [batch_size, dim]
        Returns:
            score: [batch_size, dim]
        """
        x = x.unsqueeze(1)  # [batch_size, 1, dim]
        score = torch.vmap(torch.func.jacrev(self.log_density))(x)  # [batch_size, 1, dim, dim]
        return score.squeeze((1, 2, 3))  # [batch_size, dim]


class SampleDensity(ABC):
    """
    Abstract class for probability densities we can sample from.
    """
    
    @abstractmethod
    def sample(self, n : int) -> torch.Tensor:
        """
        Gives n sample of density. 

        Args:
            n (int): amount of datapoints to sample

        Returns:
            torch.Tensor: output samples [batch_size, dim]
        """
        pass