from .base_models import *
import random
from .guidance_models import *

"""Basic Compositional SDEs/Models/Ideas etc. allowing for sampling from the product of distributions where  we have already learned models"""


"""Approximating the Product via summing scores"""
class CompLangevin(SDE):
    def __init__(self, models : list[nn.Module],  sigma : float, alpha = Alpha, beta = Beta):
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
    
"""General Compositonal Langevin using Guidance for each model, roughly based on https://arxiv.org/abs/2206.01714
in the implementation we also allow for multiple different score models, i am not sure if that is any useful in the conditional case as implement, maybe adaptable to that setting though"""
class ProductGuidanceLangevin(SDE):
    def __init__(
        self,
        models: list[nn.Module],
        alpha: Alpha,
        beta: Beta,
        sigma: float,
        null_index : int,
        model_type="score",
        guidance_scales : list[float] = None,  # 0.0 = unconditional; >0 = guided
    ):
        super().__init__()
        self.models = models
        self.beta = beta
        self.alpha = alpha
        self.sigma = sigma
        self.type = model_type
        self.null_index = null_index
        if guidance_scales is None:
            self.guidance_scales = [0.0] * len(models)
        else:
            self.guidance_scales = guidance_scales

    def guided_score(self, x_t: torch.Tensor, t: torch.Tensor, y_index:  list[torch.Tensor] | None) -> torch.Tensor:
        y_null = torch.full((x_t.shape[0],), fill_value= self.null_index, dtype=torch.long).to(device=x_t.device)
        if y_index is None:
            #scores = [model(x_t,t) for model in self.models]
            #score = sum(scores)  
            return self.models[0](x_t, t, y_null) # unconditional
        null_indices = [torch.full_like(t, self.null_index) for t in y_index]
        """In the paper they considered only the same model and used the guidance, so we assume that all models are the same, todo is work out if its useful to have different ones"""
        uncond = self.models[0](x_t, t, y_null)
        
        guided_scores =  [self.guidance_scales[i] * (self.models[i](x_t,t,null_indices[i]) - self.models[i](x_t, t, y_null)) for i in range(len(self.models))]
        return uncond + sum(guided_scores)

    def drift(self, x_t: torch.Tensor, t: torch.Tensor, y_index: list[torch.Tensor] | None = None) -> torch.Tensor:
        score = self.guided_score(x_t, t, y_index)
        beta_t = self.beta(t)
        beta_dt = self.beta.dt(t)
        alpha_t = self.alpha(t)
        alpha_dt = self.alpha.dt(t)

        if self.type == "noise":
            score = score / -beta_t

        return ((beta_t**2 * alpha_dt / alpha_t - beta_dt * beta_t + self.sigma**2 / 2) * score + (alpha_dt / alpha_t) * x_t)

    def diffusion(self, x_t: torch.Tensor, t: torch.Tensor, y_index: list[torch.Tensor] | None = None) -> torch.Tensor:
        return self.sigma * torch.randn_like(x_t)
    
    
    
"""Other version with different scaling"""
class ProductGuidanceLangevin2(SDE):
    def __init__(
        self,
        models: list[nn.Module],
        alpha: Alpha,
        beta: Beta,
        sigma: float,
        model_type="score",
        guidance_scales : list[float] = None,  # 0.0 = unconditional; >0 = guided
    ):
        super().__init__()
        self.models = models
        self.beta = beta
        self.alpha = alpha
        self.sigma = sigma
        self.type = model_type
        if guidance_scales is None:
            self.guidance_scales = [0.0] * len(models)
        else:
            self.guidance_scales = guidance_scales

    def guided_score(self, x_t: torch.Tensor, t: torch.Tensor, y:  list[torch.Tensor] | None) -> torch.Tensor:
        # Create "null" conditioning (zeros for now, maybe some other also useful?)
        y_null = torch.zeros_like(x_t)
        if y is None:
            #scores = [model(x_t,t) for model in self.models]
            #score = sum(scores)  
            return self.models[0](x_t, t, y_null) # unconditional
        
        """In the paper they considered only the same model and used the guidance, so we assume that all models are the same, todo is work out if its useful to have different ones"""
        uncond = self.models[0](x_t, t, y_null)
        
        weighted_uncond_sum = sum([(1- self.guidance_scales[i]) * self.models[i](x_t, t, y_null) for i in range(len(self.models))])
        
        guided_scores =  [self.guidance_scales[i] * (self.models[i](x_t,t,y[i])) for i in range(len(self.models))]
        return weighted_uncond_sum + sum(guided_scores)

    def drift(self, x_t: torch.Tensor, t: torch.Tensor, y: list[torch.Tensor] | None = None) -> torch.Tensor:
        assert len(self.models) == len(y), "Number of models and conditions must match"
        score = self.guided_score(x_t, t, y)
        beta_t = self.beta(t)
        beta_dt = self.beta.dt(t)
        alpha_t = self.alpha(t)
        alpha_dt = self.alpha.dt(t)

        if self.type == "noise":
            score = score / -beta_t

        return ((beta_t**2 * alpha_dt / alpha_t - beta_dt * beta_t + self.sigma**2 / 2) * score + (alpha_dt / alpha_t) * x_t)

    def diffusion(self, x_t: torch.Tensor, t: torch.Tensor, y: list[torch.Tensor] | None = None) -> torch.Tensor:
        return self.sigma * torch.randn_like(x_t)
    
    
class ProductGuidanceLangevin2(SDE):
    def __init__(
        self,
        models: list[nn.Module],
        alpha: Alpha,
        beta: Beta,
        sigma: float,
        null_index : int,
        model_type="score",
        guidance_scales : list[float] = None,  # 0.0 = unconditional; >0 = guided
    ):
        super().__init__()
        self.models = models
        self.beta = beta
        self.alpha = alpha
        self.sigma = sigma
        self.type = model_type
        self.null_index = null_index
        if guidance_scales is None:
            self.guidance_scales = [0.0] * len(models)
        else:
            self.guidance_scales = guidance_scales

    def guided_score(self, x_t: torch.Tensor, t: torch.Tensor, y_index:  list[torch.Tensor] | None) -> torch.Tensor:
        y_null = torch.full((x_t.shape[0],), fill_value= self.null_index, dtype=torch.long).to(device=x_t.device)
        if y_index is None:
            #scores = [model(x_t,t) for model in self.models]
            #score = sum(scores)  
            return self.models[0](x_t, t, y_null) # unconditional
        null_indices = [torch.full_like(t, self.null_index) for t in y_index]
        """In the paper they considered only the same model and used the guidance, so we assume that all models are the same, todo is work out if its useful to have different ones"""
        uncond = self.models[0](x_t, t, y_null)
        
        guided_scores =  [self.guidance_scales[i] * (self.models[i](x_t,t,null_indices[i]) - self.models[i](x_t, t, y_null)) for i in range(len(self.models))]
        return uncond + sum(guided_scores)

    def drift(self, x_t: torch.Tensor, t: torch.Tensor, y_index: list[torch.Tensor] | None = None) -> torch.Tensor:
        score = self.guided_score(x_t, t, y_index)
        beta_t = self.beta(t)
        beta_dt = self.beta.dt(t)
        alpha_t = self.alpha(t)
        alpha_dt = self.alpha.dt(t)

        if self.type == "noise":
            score = score / -beta_t

        return ((beta_t**2 * alpha_dt / alpha_t - beta_dt * beta_t + self.sigma**2 / 2) * score + (alpha_dt / alpha_t) * x_t)

    def diffusion(self, x_t: torch.Tensor, t: torch.Tensor, y_index: list[torch.Tensor] | None = None) -> torch.Tensor:
        return self.sigma * torch.randn_like(x_t)