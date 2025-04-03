from abc import ABC, abstractmethod
import torch
from base_probability import ImplicitDensity, ExplicitDensity, Gaussian, Alpha, Beta
from base_ode import ODE, SDE

"""Basic abstract classes for basic flows and probability. This is partially adapted from MIT CS 6.S184 by Peter Holderrieth and Ezra Erives."""

class ConditionalProbabilityPath(torch.nn.Module, ABC):
    """
    Abstract base class for conditional probability paths.
    """
    def __init__(self, p_init: ImplicitDensity, p_data: ImplicitDensity):
        super().__init__()
        self.p_init = p_init
        self.p_data = p_data

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
    def __init__(self, p_data: ImplicitDensity, alpha: Alpha, beta: Beta, device = 'cuda'):
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
