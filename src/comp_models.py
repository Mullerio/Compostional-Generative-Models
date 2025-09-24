from src.ode_solvers import *
from .base_models import *
import random
from .guidance_models import *
from .schedules import *
from .path_lib import *

"""Basic Compositional SDEs/Models/Ideas etc. allowing for sampling from the product of distributions where  we have already learned models"""

class CompLangevin(SDE):
    """Approximating the Product via summing scores"""
    def __init__(self, models : list[nn.Module], sigma : float, density = False, alpha = Alpha, beta = Beta):
        super().__init__()
        self.models = models
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.density = density
        
    def drift(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.density:
            # make sure input requires grad
            x_t_grad = x_t.detach().clone().requires_grad_(True)

            scores = []
            with torch.enable_grad():  
                for model in self.models:
                    log_density = model(x_t_grad, t)  
                    grad = torch.autograd.grad(
                        outputs=log_density.sum(),   
                        inputs=x_t_grad,
                        create_graph=False,
                        retain_graph=True
                    )[0]
                    scores.append(grad)
            score = sum(scores).detach()  # detach after grads are computed
        else:
            scores = [model(x_t, t) for model in self.models]
            score = sum(scores)

        beta_t = self.beta(t)
        beta_dt = self.beta.dt(t)
        alpha_t = self.alpha(t)
        alpha_dt = self.alpha.dt(t)

        return ((beta_t**2 * alpha_dt/alpha_t - beta_dt * beta_t + self.sigma**2 / 2) * score
                + alpha_dt/alpha_t * x_t)
    
    def diffusion(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        
        return self.sigma * torch.randn_like(x_t)
    

def sample_CompLangevin(models, n, path : GaussianConditionalProbabilityPath, device = "cuda", density = False):
    sde = CompLangevin(models=models, sigma = 1, density = density, alpha = path.alpha, beta=path.beta)
    
    comp_solver = EulerSDESolver(sde)

    comp_sampler = Sampler(comp_solver)
    
    steps= torch.linspace(0.01, 1, n).view(1,-1,1).expand(n,-1,1).to(device) 
    samples = comp_sampler.sample_without_traj(path.p_init.sample(n), steps)
    return samples


def sample_UnionLangevin(models, n, path: GaussianConditionalProbabilityPath, device="cuda", weights=None, density=False):
    num_models = len(models)
    
    if weights is None:
        weights = [1.0 / num_models] * num_models
    
    weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
    
    model_choices = torch.multinomial(weight_tensor, n, replacement=True)  # [n] indices of models
    model_counts = torch.bincount(model_choices, minlength=num_models)  # [num_models]
    
    all_samples = []
    current_idx = 0
    
    for model_idx, count in enumerate(model_counts):
        if count > 0:
            # Use the sample_CompLangevin function with only the selected model
            samples = sample_CompLangevin([models[model_idx]], count.item(), path, device, density)
            all_samples.append(samples)
    
    # Concatenate all samples
    if all_samples:
        concatenated_samples = torch.cat(all_samples, dim=0)
        
        # Reorder samples to match original random selection order
        # Create inverse permutation to restore original order
        reorder_indices = torch.empty_like(model_choices)
        current_pos = 0
        for model_idx, count in enumerate(model_counts):
            if count > 0:
                # Find positions where this model was selected
                mask = (model_choices == model_idx)
                positions = torch.where(mask)[0]
                reorder_indices[positions] = torch.arange(current_pos, current_pos + count, device=device)
                current_pos += count
        
        # Reorder to match original random selection
        final_samples = concatenated_samples[reorder_indices]
        return final_samples    
    
def DensityCompSample(models, n, path: GaussianConditionalProbabilityPath, 
                      n_importance, device="cuda", proposal_density=None, true_density = False, density=True):
    """
    Sample using importance sampling with CompLangevin models.
    
    Args:
        models: List of models to evaluate densities
        n: Number of final samples desired
        path: GaussianConditionalProbabilityPath object
        proposal_density: Proposal density to sample from
        n_importance: Minimum number of proposal samples to use
        device: Device to run on
        density: Whether to use density mode for CompLangevin
    """
    if n < n_importance:
        amount_proposal_samples = n_importance
    else: 
        amount_proposal_samples = n
        
    with torch.no_grad():
        # Sample from proposal using CompLangevin

        if true_density: 
            #assume it is density
            x = proposal_density.sample(amount_proposal_samples).to(device)  # proposal [n, dim]
            log_q = proposal_density.log_density(x).squeeze()  # [n]
        else:
            x = sample_CompLangevin([proposal_density], amount_proposal_samples, path, device, density)  # proposal [n, dim]
            # Create time tensor with same batch dimension as x and proper shape for concatenation
            t = torch.ones(amount_proposal_samples, 1, device=device)  # [batch_size, 1] tensor of ones

            log_q = proposal_density(x, t)  # [n]
            
        t = torch.full((n, 1), 0.99, device=device)  # [batch_size, 1] tensor of ones
     
        log_p_sum = sum(model(x, t) for model in models)  # [n] 
        log_weights = log_p_sum - log_q 
        weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))  # normalizing weight, just scalers

        idx = torch.multinomial(weights, n, replacement=True)  # n indices chosen from distribution over [0,...,len(weights)-1] given by weights, where each index can appear more than once 
        return x[idx]


    
    
"""General Compositional Langevin using Guidance for each model, roughly based on https://arxiv.org/abs/2206.01714
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
        """In the paper they considered only the same model and used the guidance, so we assume that all models are the same, todo is work out if its useful to have different ones"""
        uncond = self.models[0](x_t, t, y_null) * (1-sum(self.guidance_scales))

        guided_scores =  [self.guidance_scales[i] * self.models[i](x_t,t,y_index[i])  for i in range(len(self.models))]
        return uncond + sum(guided_scores)

    def drift(self, x_t: torch.Tensor, t: torch.Tensor, y_index: list[torch.Tensor] | None = None) -> torch.Tensor:
        score = self.guided_score(x_t, t, y_index)
        beta_t = self.beta(t)
        beta_dt = self.beta.dt(t)
        alpha_t = self.alpha(t)
        alpha_dt = self.alpha.dt(t)

        if self.type == "noise":
            score = score / -beta_t

        return (beta_t**2 * alpha_dt/alpha_t - beta_dt* beta_t+ self.sigma**2/2 ) * score+ alpha_dt/alpha_t *x_t

    def diffusion(self, x_t: torch.Tensor, t: torch.Tensor, y_index: list[torch.Tensor] | None = None) -> torch.Tensor:
        return self.sigma * torch.randn_like(x_t)