from sklearn.datasets import make_moons, make_swiss_roll
import matplotlib.pyplot as plt
from .prob_lib import *
from typing import Optional
import numpy as np
import seaborn as sns
from .vis import *

"""
Basic Toy Datasets for Experimentation, mostly in 2D
"""

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


#Gaussian and Multivariate Gaussian are originally taken from MIT CS 6.S184 by Peter Holderrieth and Ezra Erives, however i have made some adjustments 
class Gaussian(torch.nn.Module, LogDensity, SampleDensity):
    """
    Multivariate Gaussian distribution. Wrapper around Multivariate Gaussian
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

class GaussianMixture(torch.nn.Module, LogDensity, SampleDensity):
    """
    Two-dimensional Gaussian mixture model. Wrapper around torch.distributions.MixtureSameFamily.
    """
    def __init__(
        self,
        means: torch.Tensor,  # nmodes x data_dim
        covs: torch.Tensor,  # nmodes x data_dim x data_dim
        weights: torch.Tensor,  # nmodes
    ):
        """
        means: shape (nmodes, 2)
        covs: shape (nmodes, 2, 2)
        weights: shape (nmodes, 1)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return torch.distributions.MixtureSameFamily(
                mixture_distribution=torch.distributions.Categorical(probs=self.weights, validate_args=False),
                component_distribution=torch.distributions.MultivariateNormal(
                    loc=self.means,
                    covariance_matrix=self.covs,
                    validate_args=False,
                ),
                validate_args=False,
            )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0, seed = 0.0
    ) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale + x_offset * torch.Tensor([1.0, 0.0])
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std ** 2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0
    ) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale + torch.Tensor([1.0, 0.0]) * x_offset
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)
    
class SwissRoll2D(SampleDensity):
    def __init__(self, device : torch.device, noise : float =0.1, scale : float =1.0, rotation : float = 0.0, offset: Optional[torch.Tensor] = None):
        self.device = device
        self.noise = noise
        self.scale = scale
        self.rotation = rotation
        self.offset = offset
    
    @property
    def dim(self) -> int:
        return 2

    def sample(self, n: int) -> torch.Tensor:
        data, _ = make_swiss_roll(n_samples=n, noise=self.noise)
        data = data[:, [0, 2]]  # project to 2D 
        x = torch.tensor(data, dtype=torch.float32)
        return apply_affine_2d(x, self.scale, self.rotation, self.offset).to(self.device)


class TwoMoons2D(SampleDensity):
    def __init__(self, device : torch.device, noise=0.1, scale=1.0, rotation=0.0, offset: Optional[torch.Tensor] = None):
        self.device = device
        self.noise = noise
        self.scale = scale
        self.rotation = rotation
        self.offset = offset

    @property
    def dim(self) -> int:
        return 2

    def sample(self, n: int) -> torch.Tensor:
        x, _ = make_moons(n_samples=n, noise=self.noise)
        x = torch.tensor(x, dtype=torch.float32)
        return apply_affine_2d(x, self.scale, self.rotation, self.offset).to(self.device)



def union_sample(SampleDensity):
    def __init__(self, densities):
        self.densities = densities
        
    @property
    def dim(self) -> int:
        return self.densities[0].dim() 
    
    def sample(self, n: int) -> torch.Tensor:
        num_densities = len(self.densities)
        choices = torch.randint(0, num_densities, size=(n,))  # [n] index of all densities to sample from

        counts = torch.bincount(choices, minlength=num_densities)  # [num_densities] how many times each density was chosen

        # Sample from each density separately
        samples = []
        for idx, count in enumerate(counts):
            if count > 0:
                samples.append(self.densities[idx].sample(count))

        return torch.cat(samples, dim=0)

    

class RectangleDataset(SampleDensity):
    def __init__(self, device : torch.device,  coords : list[tuple[float, float]], rotation=0.0):
        self.device = device
        self.coords = coords
        self.rotation = rotation
        if len(coords) >= 3 and rotation != 0.0:
            raise ValueError("rotation of more than 2d datasets not implemented")

    @property
    def dim(self) -> int:
        return len(self.coords)

    def sample(self, n: int) -> torch.Tensor:
        boundaries = []
        for coord in self.coords:
            x = torch.empty(n).uniform_(coord[0], coord[1])
            boundaries.append(x)
        if self.dim > 2: 
            return torch.stack(boundaries, dim = 1)
        return apply_affine_2d(torch.stack(boundaries,  dim=1), rotation=self.rotation).to(self.device)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gmm = GaussianMixture.symmetric_2D(nmodes=5,std = 1)
    swiss = SwissRoll2D(device, noise=0.2, scale=(0.5, 0.25), rotation=0)
    moons = TwoMoons2D(device, noise=0.1, scale=2.0, offset=(1.0, -0.5))

    samples = swiss.sample(1000)
    plot_samples(samples, title="Swiss Roll",contour=False,save_path="swissrole_plot.png")

    samples = moons.sample(1000)
    plot_samples(samples, title="Two Moons",contour=False,save_path="moons_plot.png")

    samples = gmm.sample(1000)
    plot_samples(samples, title="Gaussian Mixture", contour=False, save_path="gmm5_plot.png")
