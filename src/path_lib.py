from abc import ABC, abstractmethod
import torch
from .prob_lib import LogDensity, SampleDensity, Alpha, Beta
from .ode_lib import ODE, SDE
from .dataset_lib import Gaussian

"""Basic abstract classes for basic flows and probability. This is partially adapted from MIT CS 6.S184 by Peter Holderrieth and Ezra Erives."""

class ConditionalProbabilityPath(torch.nn.Module, ABC):
    """
    Abstract base class for conditional probability paths.
    """
    def __init__(self, p_init: SampleDensity, p_data: SampleDensity, device = 'cuda'):
        super().__init__()
        self.p_init = p_init
        self.p_data = p_data
        self.device = device
            

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = int p_t(x|z) p(z)
        Args:
            t: time (n, 1)
        Returns:
            x: samples from p_t(x), (n, dim)
        """
        n = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z = self.sample_conditioning_variable(n) # [n, dim]
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # [n, dim]
        return x

    @abstractmethod
    def sample_conditioning_variable(self, n: int) -> torch.Tensor:
        """
        Samples the conditioning variable z
        Args:
            n (int): the number of samples
        Returns:
            z: sample from p(z), [n, dim]
        """
        pass
    
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            z (torch.Tensor): conditioning variable [n, dim]
            t (torch.Tensor): time [n, 1]
        Returns:
            x (torch.Tensor): sample from p_t(x|z), [n, dim]
        """
        pass
        
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            x: position variable [n, dim]
            z: conditioning variable [n, dim]
            t: time [n, 1]
        Returns:
            - conditional_vector_field: conditional vector field [n, dim]
        """ 
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            x: position variable (n, dim)
            z: conditioning variable (n, dim)
            t: time (n, 1)
        Returns:
            conditional_score: conditional score (n, dim)
        """ 
        pass
    
    
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    Gaussian Path following N(alpha_t * z, beta_t**2 * I_d)
    """
    def __init__(self, p_data: SampleDensity, alpha: Alpha, beta: Beta, device = 'cuda'):
        p_simple = Gaussian.isotropic(p_data.dim, 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
    def sample_conditioning_variable(self, n: int) -> torch.Tensor:
        """
        Samples the conditioning variable z ~ p_data(x)
        Args:
            - n: the number of samples
        Returns:
            - z: samples from p(z), (n, dim)
        """
        return self.p_data.sample(n)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
        Args:
            - z: conditioning variable [n, dim]
            - t: time (n, 1)
        Returns:
            - x: samples from p_t(x|z), (n, dim)
        """
        
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z, device=self.device)        
                
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Note: Only defined on t in [0,1)
        Args:
            - x: position variable (n, dim)
            - z: conditioning variable (n, dim)
            - t: time (n, 1)
        Returns:
            - conditional_vector_field: conditional vector field (n, dim)
        """ 
        return (self.alpha.dt(t) - self.beta.dt(t)/self.beta(t) * self.alpha(t)) * z + (self.beta.dt(t)/self.beta(t)) * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
        Note: Only defined on t in [0,1)
        Args:
            - x: position variable (n, dim)
            - z: conditioning variable (n, dim)
            - t: time (n, 1)
        Returns:
            - conditional_score: conditional score (n, dim)
        """ 
        return (self.alpha(t) *z - x)/self.beta(t)**2
    
class ConditionalVectorFieldODE(ODE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.path = path
        self.z = z

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z)
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(x,z,t)
    

class ConditionalVectorFieldSDE(SDE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor, sigma: float):
        """
        Args:
            path: the ConditionalProbabilityPath object to which this vector field corresponds
            z: the conditioning variable, (1, ...)
        """
        super().__init__()
        self.path = path
        self.z = z
        self.sigma = sigma

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z)
        Args:
            x: state at time t, shape [bs, dim]
            t: time, shape [bs,.]
        Returns:
            u_t(x|z): shape [batch_size, dim]
        """
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(x,z,t) + 0.5 * self.sigma**2 * self.path.conditional_score(x,z,t)

    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma * torch.randn_like(x)


class LinearConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_simple: SampleDensity, p_data: SampleDensity):
        super().__init__(p_simple, p_data)

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z ~ p_data(x)
        Args:
            num_samples (int) : the number of samples
        Returns:
            z (torch.Tensor): samples from p(z), [num_samples, ]
        """
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples the random variable x_t = (1-t) x_0 + tz
        Args:
            z: conditioning variable (num_samples, dim)
            t: time (num_samples, 1)
        Returns:
            x: samples from p_t(x|z), (num_samples, dim)
        """
        x0 = self.p_simple.sample(z.shape[0])
        return (1 - t) * x0 + t * z
        
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field of linear path u_t(x|z) = (z - x) / (1 - t)
        Note: Only defined on t in [0,1)
        Args:
            x: position variable [num_samples, dim]
            z: conditioning variable [num_samples, dim]
            t: time [num_samples, 1]
        Returns:
            conditional_vector_field: conditional vector field (num_samples, dim)
        """ 
        return (z - x) / (1 - t)

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Not known for Linear Conditional Probability Paths
        """ 
        raise NotImplementedError("Conditional Score is not known for Linear Probability Paths")
