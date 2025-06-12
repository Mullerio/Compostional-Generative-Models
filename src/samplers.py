from .prob_lib import * 
from .ode_lib import *
from .schedules import *

class AnnealedProduct(SDE):
    """
    Annealed Importance Sampling to sample from the product of densities, using alpha that goes from 0 to 1 to interpolate between proposal and target densities     
    DOES NOT WORK TOO WELL NEEDS CORRECTION TERM ?
    """
    def __init__(self, densities,  proposal_density, alpha : Alpha, sigma : float):
        super().__init__()
        if not isinstance(proposal_density, SampleDensity) or not isinstance(proposal_density, LogDensity):
            raise TypeError(f"Proposal Density must implement both SampleDensity and LogDensity")
        for d in densities:
            if not isinstance(d, SampleDensity) or not isinstance(d, LogDensity):
                raise TypeError(f"{d} must implement both SampleDensity and LogDensity")
            
        self.densities = densities
        self.proposal_density = proposal_density
        self.alpha = alpha
        self.sigma = sigma
        
    def drift(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        log_target = [density.score(x_t) for density in self.densities]
        target = sum(log_target)
        first = self.alpha(t) * target           
        second = (1-self.alpha(t)) * self.proposal_density.score(x_t) 
        return self.sigma**2/2 * (first + second)
    
    def diffusion(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        
        return self.sigma * torch.randn_like(x_t)