import torch
from abc import ABC, abstractmethod

class LogDensity(ABC):
    """
    Abstract class for probability densities where we know log density and its score. 
    """
    @property
    @abstractmethod
    def log_prob(self, x : torch.Tensor) -> torch.Tensor:
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


class Gaussian(torch.nn.Module, LogDensity, SampleDensity):
    """
    Multivariate Gaussian distribution, wrapper around Multivariate Gaussian
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        mean: shape [dim,]
        cov: shape [dim,dim]
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self):
        return torch.distributions.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))
        
    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)