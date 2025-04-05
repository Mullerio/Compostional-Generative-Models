from sklearn.datasets import make_moons, make_swiss_roll
import matplotlib.pyplot as plt
from prob_lib import *
from typing import Optional
from math import pi

"""
Basic Toy Datasets for Experimentation, mostly in 2D
"""

def plot_samples(samples: torch.Tensor, title="Samples"):
    samples = samples.detach().cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.6)
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.show()

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

class GaussianMixture(torch.nn.Module, LogDensity, SampleDensity):
    """
    Two-dimensional Gaussian mixture model, and is a Density and a Sampleable. Wrapper around torch.distributions.MixtureSameFamily.
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



#todo, adapt other datasets to style of the ones from above
class GaussianMixture2D(SampleDensity):
    def __init__(self,device : torch.device, mus : torch.Tensor, std=0.05, scale=1.0, rotation=0.0, offset: Optional[torch.Tensor] = None):
        super().__init__()
        self.mus = mus
        self.device = device
        self.std = std
        self.scale = scale
        self.rotation = rotation
        self.offset = offset
        self.k = len(mus)

    def sample(self, n: int) -> torch.Tensor:
        indices = torch.randint(0, self.k, (n,))
        chosen_mus = self.mus[indices]
        noise = torch.randn(n, 2) * self.std
        x = chosen_mus + noise
        return apply_affine_2d(x, self.scale, self.rotation, self.offset)


class SwissRoll2D(SampleDensity):
    def __init__(self, device : torch.device, noise=0.1, scale=1.0, rotation=0.0, offset: Optional[torch.Tensor] = None):
        self.device = device
        self.noise = noise
        self.scale = scale
        self.rotation = rotation
        self.offset = offset

    def sample(self, n: int) -> torch.Tensor:
        data, _ = make_swiss_roll(n_samples=n, noise=self.noise)
        data = data[:, [0, 2]]  # project to 2D 
        x = torch.tensor(data, dtype=torch.float32)
        return apply_affine_2d(x, self.scale, self.rotation, self.offset)


class TwoMoons2D(SampleDensity):
    def __init__(self, device : torch.device, noise=0.1, scale=1.0, rotation=0.0, offset: Optional[torch.Tensor] = None):
        self.device = device
        self.noise = noise
        self.scale = scale
        self.rotation = rotation
        self.offset = offset

    def sample(self, n: int) -> torch.Tensor:
        x, _ = make_moons(n_samples=n, noise=self.noise)
        x = torch.tensor(x, dtype=torch.float32)
        return apply_affine_2d(x, self.scale, self.rotation, self.offset)
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'

gm = GaussianMixture2D(device, mus=torch.tensor([[-1, -1], [1, 1], [-1, 1], [1, -1]]), std=0.1, scale=1.5, rotation=0.5)
swiss = SwissRoll2D(device, noise=0.2, scale=(0.5, 0.25), rotation=pi/2)
moons = TwoMoons2D(device, noise=0.1, scale=2.0, offset=(1.0, -0.5))

samples = gm.sample(1000)
plot_samples(samples, title="Gaussian Mixture")

samples = swiss.sample(1000)
plot_samples(samples, title="Swiss Roll")

samples = moons.sample(1000)
plot_samples(samples, title="Two Moons")