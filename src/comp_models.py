from .base_models import *


"""Basic Compositional SDEs/Models/Ideas etc. allowing for sampling from the product of distributions where  we have already learned models"""
class CompLangevin(SDE):
    def __init__(self, models : list[BasicMLP],  sigma : float, alpha = Alpha, beta = Beta):
        super().__init__()
        self.models = models
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        
    def drift(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        scores = [model(x_t,t) for model in self.models]
        score = sum(scores)
        beta_t = self.beta(t)
        beta_dt = self.beta.dt(t)
        alpha_t = self.alpha(t) 
        alpha_dt = self.alpha.dt(t)

        return  (beta_t**2 * alpha_dt/alpha_t - beta_dt* beta_t+ self.sigma**2/2 ) * score+ alpha_dt/alpha_t *x_t
    
    def diffusion(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        
        return self.sigma * torch.randn_like(x_t)