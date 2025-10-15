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
    
    steps= torch.linspace(0.05, 1, n).view(1,-1,1).expand(n,-1,1).to(device) 
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
    
def DensityCompSampleNormalized(models, n, path, n_importance, device="cuda",
                                proposal=None, true_density=False, t_final=0.99):
    """
    Importance sampling to estimate the normalization constant (Z) of a product distribution
    and sample from the normalized product of time-dependent unnormalized log densities.
    
    Target: p(x) = (1/Z) * ∏_i f_i(x, t_final) where f_i are unnormalized densities
    Method: Use importance sampling with proposal q(x) to estimate Z and sample from p(x)

    Args:
        models: List of learned models outputting log unnormalized densities log f_i(x, t)
        n: Number of desired output samples
        path: GaussianConditionalProbabilityPath (provides proposal distribution and schedules)
        n_importance: Number of proposal samples for importance sampling estimation
        device: torch device
        proposal: Custom proposal density (if None, uses path.p_init or first model)
        true_density: If True, proposal has .sample() and .log_density() methods
        t_final: Time at which to evaluate the learned densities (typically 0.99 or 1.0)

    Returns:
        samples: [n, dim] tensor of samples from normalized product p(x)
        logZ: scalar estimate of log normalization constant log(Z)
        weights: [n_importance] tensor of normalized importance weights
        ess: scalar effective sample size (diagnostic for weight quality)
    """
    M = max(n, n_importance)
    
    with torch.no_grad():
        # --- Step 1: Sample from proposal distribution ---
        if proposal is not None:
            if true_density:
                # Proposal is a true density object with .sample() and .log_density()
                x = proposal.sample(M).to(device)
                log_q = proposal.log_density(x).squeeze()
            else:
                # Proposal is a learned model - use CompLangevin to sample
                x = sample_CompLangevin([proposal], M, path, device, density=True)
                t_prop = torch.full((M, 1), t_final, device=device)
                log_q = proposal(x, t_prop).squeeze()
        else:
            # No proposal specified - use path's initial distribution (typically Gaussian)
            x = path.p_init.sample(M).to(device)
            log_q = path.p_init.log_density(x).squeeze()
        
        # --- Step 2: Evaluate unnormalized log target density ---
        # Target: log f(x) = ∑_i log f_i(x, t_final)
        t_eval = torch.full((M, 1), t_final, device=device)
        
        # Evaluate each model at the final time and sum log densities
        log_densities = []
        for model in models:
            log_f_i = model(x, t_eval).squeeze()  # [M] - log f_i(x, t_final)
            log_densities.append(log_f_i)
        
        log_f_total = torch.stack(log_densities, dim=0).sum(dim=0)  # [M] - ∑_i log f_i(x, t_final)
        
        # --- Step 3: Compute importance weights and estimate normalization constant ---
        # Unnormalized importance weights: w̃_i = f(x_i) / q(x_i)  
        # In log space: log w̃_i = log f(x_i) - log q(x_i)
        log_w_unnorm = log_f_total - log_q  # [M]
        
        # Estimate log normalization constant: log Z ≈ log(1/M * ∑_i w̃_i)
        # Numerically stable: log Z = logsumexp(log w̃) - log(M)
        logZ = torch.logsumexp(log_w_unnorm, dim=0) - torch.log(torch.tensor(M, device=device, dtype=log_w_unnorm.dtype))
        
        # --- Step 4: Normalize importance weights ---
        # Normalized weights: w_i = w̃_i / ∑_j w̃_j
        # In log space: log w_i = log w̃_i - logsumexp(log w̃)
        log_w_norm = log_w_unnorm - torch.logsumexp(log_w_unnorm, dim=0)
        w = torch.exp(log_w_norm)  # [M] - normalized weights
        
        # Ensure weights sum to 1 (numerical stability)
        w = w / torch.sum(w)
        
        # --- Step 5: Resample according to importance weights ---
        # This gives us samples approximately distributed as p(x) = f(x)/Z
        try:
            idx = torch.multinomial(w, n, replacement=True)
            samples = x[idx]
        except:
            # Fallback if multinomial fails (e.g., all weights are 0)
            print("Warning: Multinomial sampling failed, using uniform resampling")
            idx = torch.randint(0, M, (n,), device=device)
            samples = x[idx]
        
        # --- Step 6: Compute diagnostics ---
        # Effective sample size: ESS = 1 / ∑_i w_i²
        ess = (1.0 / torch.sum(w**2)).item()
        
        return samples, logZ.item(), w, ess


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


def AnnealedImportanceSampling(models, n, path: GaussianConditionalProbabilityPath, 
                              n_steps=100, device="cuda", density=True, 
                              proposal_sampler=None, beta_schedule="linear"):
    """
    Annealed Importance Sampling for compositional generation using learned diffusion models.
    
    Uses AIS to sample from the product: p_target ∝ p_model1 * p_model2
    by annealing from a simple proposal to the complex product.
    Uses the SAME α(t), β(t) schedules as used during training.
    
    Args:
        models: List of learned models (trained diffusion/energy models)
        n: Number of samples to generate
        path: Gaussian path for the models (contains α, β schedules)
        n_steps: Number of annealing steps
        device: Device to run on
        density: Whether models output log densities (True) or need gradient computation
        proposal_sampler: Optional custom proposal (defaults to path.p_init)
        beta_schedule: Annealing schedule ("linear", "geometric", or "cosine")
    """
    
    # Setup annealing schedule for importance sampling (different from path β!)
    if beta_schedule == "linear":
        ais_betas = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    elif beta_schedule == "geometric":
        ais_betas = torch.logspace(-3, 0, n_steps + 1, device=device)
    elif beta_schedule == "cosine":
        t = torch.linspace(0, 1, n_steps + 1, device=device)
        ais_betas = 0.5 * (1 - torch.cos(t * torch.pi))
    else:
        raise ValueError("Unknown beta_schedule")
    
    # Get the EXACT alpha and beta schedule objects from the path (same as CompLangevin)
    alpha = path.alpha  # This is the LinearAlpha() instance
    beta = path.beta    # This is the SquareRootBeta() instance
    
    # Initialize from proposal distribution (use path's initial distribution)
    if proposal_sampler is None:
        x = path.p_init.sample(n).to(device)
        log_proposal = path.p_init.log_density(x).squeeze()
    else:
        x = proposal_sampler.sample(n).to(device)
        log_proposal = proposal_sampler.log_density(x).squeeze()
    
    # Initialize importance weights
    log_weights = torch.zeros(n, device=device)
    
    # Use EXACT same time schedule as sample_CompLangevin: from 0.01 to 1.0
    time_steps = torch.linspace(0.01, 1.0, n_steps + 1, device=device)
    
    # Main AIS loop - follow the reverse SDE process
    for step in range(n_steps):
        ais_beta_prev = ais_betas[step]
        ais_beta_curr = ais_betas[step + 1]
        delta_ais_beta = ais_beta_curr - ais_beta_prev
        
        # Current time in the diffusion process (forward: 0.01 → 1.0)
        t_curr = time_steps[step]
        t_next = time_steps[step + 1] if step < n_steps - 1 else time_steps[step]
        
        # Prepare time tensor for model evaluation
        t_tensor = torch.full((n, 1), t_curr, device=device)
        
        with torch.no_grad():
            # Compute target log densities at current time step (following training)
            if density:
                # Models output log densities directly
                log_densities = [model(x, t_tensor).squeeze() for model in models]
            else:
                # Need to compute gradients for score-based models
                x_grad = x.detach().clone().requires_grad_(True)
                log_densities = []
                
                with torch.enable_grad():
                    for model in models:
                        log_p = model(x_grad, t_tensor)
                        log_densities.append(log_p.squeeze())
                
                log_densities = [ld.detach() for ld in log_densities]
                x = x_grad.detach()
            
            # Sum log densities for product
            log_target_sum = sum(log_densities)
            
            # Update importance weights
            log_weights += delta_ais_beta * log_target_sum
        
        # Langevin MCMC step using the path's SDE (like in training)
        if step < n_steps - 1:
            dt = t_next - t_curr  # Positive dt since we go forward in time
            
            # Use EXACT same alpha/beta calls as CompLangevin.drift()
            alpha_t = alpha(t_tensor)    # Same as self.alpha(t) in CompLangevin
            beta_t = beta(t_tensor)      # Same as self.beta(t) in CompLangevin
            alpha_dt = alpha.dt(t_tensor)  # Same as self.alpha.dt(t) in CompLangevin
            beta_dt = beta.dt(t_tensor)    # Same as self.beta.dt(t) in CompLangevin
            
            x_mcmc = x.detach().clone().requires_grad_(True)
            
            with torch.enable_grad():
                # Compute scores from models at current time
                if density:
                    # Get log densities and compute gradients
                    log_densities_grad = []
                    for model in models:
                        log_p = model(x_mcmc, t_tensor)
                        grad = torch.autograd.grad(
                            outputs=log_p.sum(),
                            inputs=x_mcmc,
                            create_graph=False,
                            retain_graph=True
                        )[0]
                        log_densities_grad.append(grad)
                    
                    # Interpolate between proposal score and target scores
                    score_target = sum(log_densities_grad)
                    score_proposal = -x_mcmc  # Gaussian proposal score
                    score_interp = ais_beta_curr * score_target + (1 - ais_beta_curr) * score_proposal
                else:
                    # Models output scores directly
                    scores = [model(x_mcmc, t_tensor) for model in models]
                    score_target = sum(scores)
                    score_proposal = -x_mcmc
                    score_interp = ais_beta_curr * score_target + (1 - ais_beta_curr) * score_proposal
            
            # SDE update using EXACT same formula as CompLangevin.drift()
            sigma = 1.0  # Same as CompLangevin 
            drift = ((beta_t**2 * alpha_dt/alpha_t - beta_dt * beta_t + sigma**2/2) * score_interp + 
                    alpha_dt/alpha_t * x_mcmc).detach()
            diffusion = sigma
            
            # Euler step
            noise = torch.randn_like(x)
            x = x + drift * abs(dt) + diffusion * torch.sqrt(torch.tensor(abs(dt))) * noise
    
    # Normalize importance weights
    log_weights_normalized = log_weights - torch.logsumexp(log_weights, dim=0)
    weights = torch.exp(log_weights_normalized)
    
    # Resample according to importance weights
    if torch.all(torch.isfinite(weights)):
        try:
            indices = torch.multinomial(weights, n, replacement=True)
            final_samples = x[indices]
        except:
            # Fallback if multinomial fails
            print("Warning: Multinomial sampling failed, returning uniform samples")
            indices = torch.randint(0, n, (n,), device=device)
            final_samples = x[indices]
    else:
        print("Warning: Invalid weights detected, returning uniform samples")
        indices = torch.randint(0, n, (n,), device=device)
        final_samples = x[indices]
    
    return final_samples


def EnergyBasedComposition(models, n, path: GaussianConditionalProbabilityPath,
                          device="cuda", density=True, mcmc_steps=100, 
                          step_size=0.01, temperature=1.0):
    """
    Energy-based composition using Langevin MCMC with proper time-varying schedules.
    Uses the SAME α(t), β(t) schedules as during training.
    
    Args:
        models: List of learned models 
        n: Number of samples
        path: Gaussian path (contains α, β schedules)
        device: Device
        density: Whether models output log densities
        mcmc_steps: Number of MCMC steps
        step_size: Base step size
        temperature: Temperature for sampling
    """
    
    # Get the EXACT alpha and beta schedule objects from the path (same as CompLangevin)
    alpha = path.alpha  # This is the LinearAlpha() instance
    beta = path.beta    # This is the SquareRootBeta() instance
    
    # Initialize from path's initial distribution (same as training)
    x = path.p_init.sample(n).to(device)
    
    # Use EXACT same time schedule as sample_CompLangevin: from 0.01 to 1.0
    time_schedule = torch.linspace(0.01, 1.0, mcmc_steps, device=device)
    
    for step in range(mcmc_steps):
        # Current time in diffusion process
        t_curr = time_schedule[step]
        t_tensor = torch.full((n, 1), t_curr, device=device)
        
        # Use EXACT same alpha/beta calls as CompLangevin.drift()
        alpha_t = alpha(t_tensor)      # Same as self.alpha(t) in CompLangevin
        beta_t = beta(t_tensor)        # Same as self.beta(t) in CompLangevin  
        alpha_dt = alpha.dt(t_tensor)  # Same as self.alpha.dt(t) in CompLangevin
        beta_dt = beta.dt(t_tensor)    # Same as self.beta.dt(t) in CompLangevin
        
        x_grad = x.detach().clone().requires_grad_(True)
        
        with torch.enable_grad():
            if density:
                # Models output log densities - compute scores via gradients
                scores = []
                for model in models:
                    log_p = model(x_grad, t_tensor)
                    grad = torch.autograd.grad(
                        outputs=log_p.sum(),
                        inputs=x_grad,
                        create_graph=False,
                        retain_graph=True
                    )[0]
                    scores.append(grad)
                total_score = sum(scores)
            else:
                # Models output scores directly
                scores = [model(x_grad, t_tensor) for model in models]
                total_score = sum(scores)
        
        # Use EXACT same SDE dynamics as CompLangevin.drift()
        sigma = 1.0  # Same sigma as CompLangevin
        drift = ((beta_t**2 * alpha_dt/alpha_t - beta_dt * beta_t + sigma**2/2) * total_score + 
                alpha_dt/alpha_t * x_grad).detach()
        diffusion = sigma
        
        # Adaptive step size based on time
        adaptive_step = step_size * (1.0 - t_curr + 0.1)  # Smaller steps as t->0
        
        # SDE update - use fixed small step size for stability
        dt = 0.001  # Small fixed step size
        noise = torch.randn_like(x)
        x = x + drift * dt + diffusion * torch.sqrt(torch.tensor(dt)) * noise
    
    return x

def IntersectionSampling(models, n_samples_per_model, path: GaussianConditionalProbabilityPath, 
                        device="cuda", ball_radius=0.5):
    """
    Simple intersection-based compositional sampling.
    
    Algorithm:
    1. Generate n_samples_per_model samples from each model independently
    2. For each sample from model1, find samples from model2 within ball_radius
    3. Return all samples that have "neighbors" in all other models
    4. This approximates the product by finding overlapping regions
    
    Args:
        models: List of learned models
        n_samples_per_model: Number of samples to generate from each model
        path: Gaussian path for sampling
        device: Device
        ball_radius: Radius for intersection ball
    
    Returns:
        Tensor of intersection samples (could be any number depending on overlap)
    """
    
    # Generate samples from each model independently
    model_samples = []
    for i, model in enumerate(models):
        print(f"Generating {n_samples_per_model} samples from model {i+1}...")
        
        # Use the same approach as your working cell #6
        sde = CompLangevin(
            models=[model], 
            sigma=1.0, 
            alpha=path.alpha, 
            beta=path.beta, 
            density=True
        )
        
        from .ode_solvers import EulerSDESolver
        from .samplers import Sampler
        
        comp_solver = EulerSDESolver(sde)
        comp_sampler = Sampler(comp_solver)
        
        # The sampler expects batch_size == num_time_steps due to the loop structure
        num_time_steps = min(n_samples_per_model, 1000)  # Cap at 1000 to avoid memory issues
        actual_batch_size = num_time_steps  # Must match for the loop to work
        
        steps = torch.linspace(0.01, 1, num_time_steps).view(1, -1, 1).expand(actual_batch_size, -1, 1).to(device)
        x_init = path.p_init.sample(actual_batch_size)
        samples = comp_sampler.sample_without_traj(x_init, steps)
        
        model_samples.append(samples.cpu())  # Move to CPU to save GPU memory
    
    print(f"Finding intersections with ball radius = {ball_radius}...")
    
    # Find intersection by checking distances between samples
    # Use model 0 as reference, find its samples that have neighbors in all other models
    reference_samples = model_samples[0]
    accepted_samples = []
    
    for i in range(reference_samples.shape[0]):
        ref_sample = reference_samples[i:i+1]  # [1, dim]
        
        # Check if this sample has neighbors in ALL other models
        has_neighbors_in_all = True
        
        for j in range(1, len(model_samples)):
            other_samples = model_samples[j]  # [n_oversample, dim]
            
            # Compute distances to all samples in model j
            distances = torch.norm(other_samples - ref_sample, dim=1)  # [n_oversample]
            min_distance = distances.min().item()
            
            # If no sample is within the ball radius, reject this reference sample
            if min_distance > ball_radius:
                has_neighbors_in_all = False
                break
        
        # If this sample has neighbors in all models, accept it
        if has_neighbors_in_all:
            accepted_samples.append(ref_sample)
            
    if len(accepted_samples) == 0:
        print(f"Warning: No intersections found with radius {ball_radius}. Try larger radius.")
        # Return empty tensor with correct shape
        return torch.empty(0, model_samples[0].shape[1]).to(device)
    
    # Return all valid intersections found - no forced count
    final_samples = torch.cat(accepted_samples, dim=0).to(device)
    
    print(f"Successfully found {final_samples.shape[0]} intersection samples!")
    return final_samples


def OversamplingIntersectionSampling(models, n, oversample_factor, path, device, ball_radius=0.5):
    """
    Original intersection sampling with oversampling to guarantee fixed output count.
    
    Args:
        models: List of trained models  
        n: Desired number of output samples (will force this count)
        oversample_factor: Multiplier for generating extra samples (e.g., 3.0 = 3x oversampling)
        path: Gaussian path for sampling
        device: Device
        ball_radius: Radius for intersection ball
    
    Returns:
        Tensor of exactly n samples (repeats samples if needed)
    """
    n_oversample = int(n * oversample_factor)
    
    # Generate samples from each model independently
    model_samples = []
    for i, model in enumerate(models):
        print(f"Generating {n_oversample} samples from model {i+1}...")
        
        # Use the same approach as your working cell #6
        sde = CompLangevin(
            models=[model], 
            sigma=1.0, 
            alpha=path.alpha, 
            beta=path.beta, 
            density=True
        )
        
        from .ode_solvers import EulerSDESolver
        from .samplers import Sampler
        
        comp_solver = EulerSDESolver(sde)
        comp_sampler = Sampler(comp_solver)
        
        # The sampler expects batch_size == num_time_steps due to the loop structure
        num_time_steps = min(n_oversample, 1000)  # Cap at 1000 to avoid memory issues
        actual_batch_size = num_time_steps  # Must match for the loop to work
        
        steps = torch.linspace(0.01, 1, num_time_steps).view(1, -1, 1).expand(actual_batch_size, -1, 1).to(device)
        x_init = path.p_init.sample(actual_batch_size)
        samples = comp_sampler.sample_without_traj(x_init, steps)
        
        model_samples.append(samples.cpu())  # Move to CPU to save GPU memory
    
    print(f"Finding intersections with ball radius = {ball_radius}...")
    
    # Find intersection by checking distances between samples
    # Use model 0 as reference, find its samples that have neighbors in all other models
    reference_samples = model_samples[0]
    accepted_samples = []
    
    for i in range(reference_samples.shape[0]):
        ref_sample = reference_samples[i:i+1]  # [1, dim]
        
        # Check if this sample has neighbors in ALL other models
        has_neighbors_in_all = True
        
        for j in range(1, len(model_samples)):
            other_samples = model_samples[j]  # [n_oversample, dim]
            
            # Compute distances to all samples in model j
            distances = torch.norm(other_samples - ref_sample, dim=1)  # [n_oversample]
            min_distance = distances.min().item()
            
            # If no sample is within the ball radius, reject this reference sample
            if min_distance > ball_radius:
                has_neighbors_in_all = False
                break
        
        # If this sample has neighbors in all models, accept it
        if has_neighbors_in_all:
            accepted_samples.append(ref_sample)
            
        # Stop early if we have enough samples
        if len(accepted_samples) >= n:
            break
    
    # Handle case where only one model (testing)
    if len(models) == 1:
        return model_samples[0][:n].to(device)
    
    if len(accepted_samples) == 0:
        print(f"Warning: No intersections found with radius {ball_radius}. Try larger radius or more oversampling.")
        # Fallback: just return some samples from model 0
        return model_samples[0][:n].to(device)
    
    # Force exact count by repeating samples if needed
    if len(accepted_samples) < n:
        print(f"Warning: Only found {len(accepted_samples)} intersections, requested {n}. Repeating samples to reach target count.")
        # Repeat samples to reach desired number
        while len(accepted_samples) < n:
            accepted_samples.extend(accepted_samples[:min(len(accepted_samples), n - len(accepted_samples))])
    
    final_samples = torch.cat(accepted_samples[:n], dim=0).to(device)
    
    print(f"Successfully generated exactly {final_samples.shape[0]} intersection samples!")
    return final_samples


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


def sample_ProductAIS(models, n, path: GaussianConditionalProbabilityPath, device="cuda"):
    """
    Convenient wrapper for AIS sampling from product of learned diffusion models.
    
    Uses the SAME time-varying α(t), β(t) schedules as during training.
    This is the main recommended method for compositional sampling.
    """
    return AnnealedImportanceSampling(
        models=models,
        n=n, 
        path=path,
        n_steps=100,  # More steps for better accuracy
        device=device,
        density=True,  # Assume models output log densities
        beta_schedule="linear"  # Linear annealing for AIS
    )


def sample_ProductMCMC(models, n, path: GaussianConditionalProbabilityPath, device="cuda"):
    """
    Convenient wrapper for MCMC sampling from product of learned models.
    
    Uses the SAME time-varying α(t), β(t) schedules as during training.
    Simpler alternative to AIS.
    """
    return EnergyBasedComposition(
        models=models,
        n=n,
        path=path,
        device=device,
        density=True,
        mcmc_steps=500,  # More steps for convergence
        step_size=0.01,  # Slightly larger base step
        temperature=0.8  # Lower temperature for sharper distribution
    )