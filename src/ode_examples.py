from ode_solvers import *

class HarmonicOscillator(ODE):
    def drift(self, x_t, t):
        return -x_t

class OrnsteinUhlenbeck(SDE):
    def __init__(self, sigma: float):
        self.sigma = sigma
        
    def drift(self, x_t, t):
        return -x_t
    
    def difussion(self, x_t, t):
        return torch.ones_like(x_t) * self.sigma

ode_solver = RungeKuttaSolver(HarmonicOscillator())
sde_solver = EulerSDESolver(OrnsteinUhlenbeck(sigma=0.5))

ode_sampler = Sampler(ode_solver)
sde_sampler = Sampler(sde_solver)

t_steps = torch.linspace(0, 1, 10).unsqueeze(-1)
x_init = torch.randn(32, 2)  # Batch of 32, 2D system

ode_traj = ode_sampler.sample_with_traj(x_init, t_steps)

sde_traj = sde_sampler.sample_with_traj(x_init, t_steps)