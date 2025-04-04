from sklearn.datasets import make_moons, make_swiss_roll
import matplotlib.pyplot as plt
from prob_lib import *
from typing import Optional
from math import pi

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


class GaussianMixture2D(SampleDensity):
    def __init__(self,device : torch.device, mus : torch.Tensor, std=0.05, scale=1.0, rotation=0.0, offset: Optional[torch.Tensor] = None):
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