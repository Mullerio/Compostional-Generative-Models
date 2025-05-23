import torch
from abc import ABC, abstractmethod
from scipy.stats import norm
import numpy as np


"""Abstract classes for probability, including schedulers for gaussian paths """


            
def rect_gauss_overlap_mass(center: torch.Tensor, std: float, rectangle: list[tuple[float, float]]) -> float:
    """
    Estimate the probability mass of a multivariate Gaussian within a rectangle.

    Args:
        center: torch.Tensor of shape [D] – mean of the Gaussian
        std: float – standard deviation (assumed same in all dimensions)
        rectangle: list of (lower, upper) tuples – one per dimension

    Returns:
        Float – estimated probability mass inside the rectangle
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


class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(self(torch.zeros(1,1)), torch.zeros(1,1))
        # Check alpha_1 = 1
        assert torch.allclose(self(torch.ones(1,1)), torch.ones(1,1))
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            t: time (num_samples, 1)
        Returns:
            alpha_t (num_samples, 1)
        """ 
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            t: time (num_samples, 1)
        Returns:
            d/dt alpha_t (num_samples, 1)
        """ 
        t = t.unsqueeze(1) # (num_samples, 1, 1)
        dt = torch.vmap(torch.functional.jacrev(self))(t) # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)
    
class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(self(torch.zeros(1,1)), torch.ones(1,1))
        # Check beta_1 = 0
        assert torch.allclose(self(torch.ones(1,1)), torch.zeros(1,1))
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """ 
        pass 

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt beta_t (num_samples, 1)
        """ 
        t = t.unsqueeze(1) # (num_samples, 1, 1)
        dt = torch.vmap(torch.functional.jacrev(self))(t) # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)
    
class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """ 
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        return torch.ones_like(t)


class LinearBeta(Beta):
    """
    Implements alpha_t = t
    """
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """ 
        return 1-t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        return -torch.ones_like(t)


class SquareRootBeta(Beta):
    """
    Implements beta_t = rt(1-t)
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """ 
        return torch.sqrt(1-t)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        return - 0.5 / (torch.sqrt(1 - t) + 1e-4)