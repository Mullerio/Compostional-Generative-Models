from ode_lib import SDE, Solver, ODE, Sampler
import torch

"""Basic ODE/SDE solvers in Torch using ode_lib"""

class RungeKuttaSolver(Solver):
    """
    4th-order Runge-Kutta solver for ODEs
    """
    def __init__(self, ode: ODE):
        super().__init__()
        self.ode = ode

    def step(self, x_t: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [batch_size, dim]
            t: [batch_size, 1]
            dt: [batch_size, 1]
        Returns:
            next_x: [batch_size, dim]
        """
        k1 = self.ode.drift(x_t, t)
        k2 = self.ode.drift(x_t + 0.5*dt*k1, t + 0.5*dt)
        k3 = self.ode.drift(x_t + 0.5*dt*k2, t + 0.5*dt)
        k4 = self.ode.drift(x_t + dt*k3, t + dt)
        
        next_x = x_t + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return next_x

class EulerODESolver(Solver):
    """
    Euler solver for ODEs
    """
    def __init__(self, ode : ODE):
        self.ode = ode

    def step(self, x_t : torch.Tensor, t : torch.Tensor, dt : torch.Tensor) -> torch.Tensor:
        
        drift = self.ode.drift(x_t,t)
        return x_t + dt * drift

class EulerSDESolver(Solver):
    """
    Euler-Maruyama solver for SDEs
    """
    def __init__(self, sde: SDE):
        super().__init__()
        self.sde = sde

    def step(self, x_t: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [batch_size, dim]
            t: [batch_size, 1]
            dt: [batch_size, 1]
        Returns:
            next_x: [batch_size, dim]
        """
        dw = torch.randn_like(x_t) * torch.sqrt(dt)
        
        drift = self.sde.drift(x_t, t) * dt
        diffusion = self.sde.difussion(x_t, t) * dw 
        
        next_x = x_t + drift + diffusion
        return next_x

# Example Usage ###########################################################

class HarmonicOscillator(ODE):
    """Simple ODE example: dx/dt = -x"""
    def drift(self, x_t, t):
        return -x_t

class OrnsteinUhlenbeck(SDE):
    """SDE example: dx_t = -x_t dt + σ dW_t"""
    def __init__(self, sigma: float):
        self.sigma = sigma
        
    def drift(self, x_t, t):
        return -x_t
    
    def difussion(self, x_t, t):
        return torch.ones_like(x_t) * self.sigma

# Create solvers
ode_solver = RungeKuttaSolver(HarmonicOscillator())
sde_solver = EulerSDESolver(OrnsteinUhlenbeck(sigma=0.5))

# Create samplers
ode_sampler = Sampler(ode_solver)
sde_sampler = Sampler(sde_solver)

# Sample trajectories
t_steps = torch.linspace(0, 1, 10).unsqueeze(-1)
x_init = torch.randn(32, 2)  # Batch of 32, 2D system

# ODE trajectory
ode_traj = ode_sampler.sample_with_traj(x_init, t_steps)

# SDE trajectory (different noise realization each time)
sde_traj = sde_sampler.sample_with_traj(x_init, t_steps)
