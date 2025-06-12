from sklearn.datasets import make_moons, make_swiss_roll
import torch.nn as nn 
import matplotlib.pyplot as plt
from .prob_lib import *
from typing import Optional
import numpy as np
import seaborn as sns
from .vis import *

"""Collection of useful dataset functions, such as weighted unions, products via importance sampling etc."""


def apply_affine_2d(x : torch.Tensor, scale=1.0, rotation=0.0, offset: Optional[torch.Tensor] = None):
    if offset is None: 
        offset = (0,0)
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    scale_matrix = torch.diag(torch.tensor(scale, dtype=torch.float32))

    rotation = torch.tensor(rotation)
    rot_matrix = torch.tensor([
        [torch.cos(rotation), -torch.sin(rotation)],
        [torch.sin(rotation),  torch.cos(rotation)]
    ], dtype=torch.float32)

    affine_matrix = rot_matrix @ scale_matrix
    return x @ affine_matrix.T + torch.tensor(offset, dtype=torch.float32)


class ProductAISSample(LogDensity, SampleDensity):
    def __init__(self, densities: list, proposal_density, alpha: Alpha, K=100, sigma=0.1, n_importance=10000):
        if not isinstance(proposal_density, SampleDensity) or not isinstance(proposal_density, LogDensity):
            raise TypeError("Proposal Density must implement both SampleDensity and LogDensity")
        for d in densities:
            if not isinstance(d, SampleDensity) or not isinstance(d, LogDensity):
                raise TypeError(f"{d} must implement both SampleDensity and LogDensity")

        self.densities = densities
        self.proposal_density = proposal_density
        self.alpha = alpha  
        self.K = K
        self.sigma = sigma
        self.n_importance = n_importance

        self.estimate_logZ()

    @property
    def dim(self) -> int:
        return self.densities[0].dim

    def interpolated_log_density(self, x, alpha_val):
        log_p_sum = sum(d.log_density(x).squeeze() for d in self.densities)
        log_q = self.proposal_density.log_density(x).squeeze()
        return (1 - alpha_val) * log_q + alpha_val * log_p_sum

    def interpolated_score(self, x, alpha_val):
        score_p = sum(d.score(x) for d in self.densities)
        score_q = self.proposal_density.score(x)
        return (1 - alpha_val) * score_q + alpha_val * score_p

    def langevin_step(self, x, alpha_val, eps):
        score = self.interpolated_score(x, alpha_val)
        noise = torch.randn_like(x)
        return x + 0.5 * eps * score + self.sigma * noise * eps**0.5

    def estimate_logZ(self):
        with torch.no_grad():
            x = self.proposal_density.sample(self.n_importance)
            log_w = torch.zeros(self.n_importance)

            for i in range(1, self.K + 1):
                t0 = torch.tensor((i - 1) / self.K) #we fix uniform steps 
                t1 = torch.tensor(i / self.K)
                alpha0 = self.alpha(t0)
                alpha1 = self.alpha(t1)

                log_pi_t0 = self.interpolated_log_density(x, alpha0)
                log_pi_t1 = self.interpolated_log_density(x, alpha1)
                log_w += log_pi_t1 - log_pi_t0

                x = self.langevin_step(x, alpha1, eps=1.0 / self.K)

            self.logZ = torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(self.n_importance, dtype=torch.float32))

    def log_density(self, x):
        log_p_sum = sum(d.log_density(x).squeeze() for d in self.densities)
        return (log_p_sum - self.logZ).unsqueeze(1)

    def sample(self, n: int):
        with torch.no_grad():
            amount_proposal_samples = max(n, self.n_importance)
            x = self.proposal_density.sample(amount_proposal_samples)
            log_w = torch.zeros(amount_proposal_samples)

            for i in range(1, self.K + 1):
                t0 = torch.tensor((i - 1) / self.K)
                t1 = t1 = torch.tensor(i / self.K)
                alpha0 = self.alpha(t0)
                alpha1 = self.alpha(t1)

                log_pi_t0 = self.interpolated_log_density(x, alpha0)
                log_pi_t1 = self.interpolated_log_density(x, alpha1)
                log_w += log_pi_t1 - log_pi_t0

                x = self.langevin_step(x, alpha1, eps=1.0 / self.K)

            weights = torch.exp(log_w - torch.logsumexp(log_w, dim=0))
            idx = torch.multinomial(weights, n, replacement=True)
            return x[idx]
        

class ProductLogSample(LogDensity, SampleDensity):
    def __init__(self, densities: list, proposal_density, n_importance: int = 10000):
        if not isinstance(proposal_density, SampleDensity) or not isinstance(proposal_density, LogDensity):
            raise TypeError(f"Proposal Density must implement both SampleDensity and LogDensity")
        for d in densities:
            if not isinstance(d, SampleDensity) or not isinstance(d, LogDensity):
                raise TypeError(f"{d} must implement both SampleDensity and LogDensity")
        
        self.densities = densities
        self.num_densities = len(densities)
        self.proposal_density = proposal_density
        self.n_importance = n_importance  # number of samples used to estimate Z

        self.estimate_logZ() # setting logZ inside the method 

    @property
    def dim(self) -> int:
        return self.densities[0].dim

    def estimate_logZ(self):
        """
        Estimate log of the normalizing constant Z using importance sampling.
        """
        with torch.no_grad():
            x = self.proposal_density.sample(self.n_importance)  # [n, dim]

            log_p_sum = sum(d.log_density(x).squeeze() for d in self.densities)
            log_q = self.proposal_density.log_density(x).squeeze()

            log_weights = log_p_sum - log_q
            self.logZ = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(self.n_importance, dtype=torch.float32))

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculating the log density of the product of densities using the normalizing constant Z.
        Args:
            x (torch.Tensor): [batch_size, 1]
        """
        log_p_sum = sum(d.log_density(x).squeeze() for d in self.densities)
        log_density = log_p_sum - self.logZ
        return log_density.unsqueeze(1)  # [batch_size, 1]

    def sample(self, n: int) -> torch.Tensor:
        """
        Sampling from the product using Importance Sampling, since we only have the log_densities currently, this is written in log terms
        
        """
        if n < self.n_importance:
            amount_proposal_samples = self.n_importance
        else: 
            amount_proposal_samples = n
        with torch.no_grad():
            x = self.proposal_density.sample(amount_proposal_samples)  # proposal [n, dim]
            log_p_sum = sum(d.log_density(x).squeeze() for d in self.densities) # [n] due to squeeze
            log_q = self.proposal_density.log_density(x).squeeze() #  [n] due to squeeze

            log_weights = log_p_sum - log_q 
            weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))  # normalizing weight, just scalers

            idx = torch.multinomial(weights, n, replacement=True) #n indices chosen from distribution over [0,...,len(weights)-1] given by weights, where each index can appear more than once 
            return x[idx]
        

class UnionLogSample(LogDensity, SampleDensity):
    """
    Weighted "union" of multiple densities/distributions that all implement Sample and LogDensity
    """
    def __init__(self, densities: list, weights: list[float] = None):
        for d in densities:
            if not isinstance(d, SampleDensity):
                raise TypeError(f"{d} does not implement SampleDensity, need both log and sample density")
            if not isinstance(d, LogDensity):
                raise TypeError(f"{d} does not implement LogDensity, need both log and sample density")     
        
        self.densities = densities
        self.num_densities = len(densities)

        #if no weights, assume uniform
        if weights is None:
            self.weights = [1.0 / self.num_densities] * self.num_densities
        else:
            assert np.allclose(sum(weights), 1)
            self.weights = weights
    @property
    def dim(self) -> int:
        return self.densities[0].dim
    
    def sample(self, n: int) -> torch.Tensor:
        weight_tensor = torch.tensor(self.weights, dtype=torch.float) #to use multinomal torch function, convert to tensor

        choices = torch.multinomial(weight_tensor, n, replacement=True)  # [n] indices of densities to sample from
        counts = torch.bincount(choices, minlength=self.num_densities) # count how much to sample from each density 

        samples = []
        for idx, count in enumerate(counts):
            if count > 0:
                samples.append(self.densities[idx].sample(count))

        return torch.cat(samples, dim=0)
    
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = []
        for idx, density in enumerate(self.densities):
            log_prob = density.log_density(x)
            log_probs.append(log_prob + torch.log(torch.tensor(self.weights[idx], dtype=torch.float32)))
        c = torch.logsumexp(torch.stack(log_probs, dim=1), dim=1)  # [batch_size, 1]
        return c



class UnionSample(SampleDensity):
    """    
    Weighted "union" of multiple densities/distributions that all implement SampleDensity (this is the class one could use on more "real" world data like images)
    """
    def __init__(self, densities: list[SampleDensity], weights: list[float] = None):
        self.densities = densities
        self.num_densities = len(densities)

        if weights is None:
            self.weights = [1.0 / self.num_densities] * self.num_densities
        else:
            assert np.allclose(sum(weights), 1)
            self.weights = weights
            
    @property
    def dim(self) -> int:
        return self.densities[0].dim

    def sample(self, n: int) -> torch.Tensor:
        weight_tensor = torch.tensor(self.weights, dtype=torch.float) #to use multinomal torch function, convert to tensor

        choices = torch.multinomial(weight_tensor, n, replacement=True)  # [n] indices of densities to sample from
        counts = torch.bincount(choices, minlength=self.num_densities) # count how much to sample from each density 

        samples = []
        for idx, count in enumerate(counts):
            if count > 0:
                samples.append(self.densities[idx].sample(count))

        return torch.cat(samples, dim=0)