import torch
from abc import ABC, abstractmethod
from tqdm import tqdm

"""Basic abstract classes for ODEs/SDEs using Torch, adapted from the labs of MIT CS 6.S184 by Peter Holderrieth and Ezra Erives."""

class ODE(ABC):
    """
    Abstract class for basic time dependent first order Ordinary differential equations given by 
    dx/dt = f(x,t) where f is some vector field, this can be written as dx_t = f(x_t, t) dt. We assume that f is given on [0,T]
    """
    @abstractmethod
    def drift(self, x_t : torch.Tensor,t : torch.Tensor) -> torch.Tensor:
        """
        Returns f(x_t,t) 
        Args:
            x_t (torch.Tensor): input at time t [batch_size, dim]
            t (torch.Tensor): time t

        Returns:
            torch.Tensor: gives drift term of ODE [batch_size, dim]
        """
        pass
    
class Solver(ABC):
    """
    Abstract Class for ODE solvers. Think Runge Kutta, Euler, Huen etc.
    """
    @abstractmethod
    def step(self, x_t : torch.Tensor, t : torch.Tensor, d_t : torch.Tensor) -> torch.Tensor: 
        """
        Performs a single discrete integration step
        Args:
            x_t (torch.Tensor): Current state [batch_size, dim]
            t (torch.Tensor): Current time [batch_size,1]
            d_t (torch.Tensor): Given time step tensor [batch_size, 1]

        Returns:
            torch.Tensor: Next state tensor [batch_size, dim]
        """
        pass
    

class Sampler(ABC):
    """
    Abstract Class for samplers of ODEs/SDEs using ODE/SDE solvers.
    """
    def __init__(self, solver : Solver):
        if solver is None: 
            raise ValueError("Give a valid Sovler.")
        self.solver = solver 
        
    def set_solver(self, new_solver : Solver):
        """Setter for dynamically changing solver. """
        self.solver = new_solver
            
    @torch.no_grad()
    def sample_with_traj(self, x_init : torch.tensor, steps : torch.Tensor, **kwargs) -> torch.Tensor: 
        """
        Sample the ODE using the given Solver of the Sampler Class and store its trajectory

        Args:
            x_init (torch.tensor): initial state at t_start [batch_size, dim]
            steps (torch.Tensor): [batch_size,number_timesteps,1]

        Returns:
            torch.Tensor: trajectory of ODE [number_timesteps, batch_size, dim]
        """    
        number_timesteps = len(steps)
        traj = torch.zeros((number_timesteps, *x_init.shape), dtype=x_init.dtype, device=x_init.device)
        traj[0] = x_init
        x = x_init
        for idx in range(len(steps) - 1):
            step_now = steps[:, idx]
            dt = steps[:, idx + 1] - steps[:, idx]
            x = self.solver.step(x, step_now, dt, **kwargs)
            traj[idx+1] = x.clone() #dont pass reference,store current x

        return traj
    
    
    
    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (bs, dim)
            - ts: timesteps, shape (bs, num_timesteps, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, num
            _timesteps, dim)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.solver.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
    
    @torch.no_grad()
    def sample_without_traj(self, x_init : torch.tensor, steps : torch.Tensor, **kwargs) -> torch.Tensor: 
        """
        Sample the ODE using the given Solver of the Sampler Class without storing the entire trajectory

        Args:
            x_init (torch.tensor): initial state at t_start [batch_size, dim]
            steps (torch.Tensor): steps[i] == timestep we sample at step i [number_timesteps]

        Returns:
            torch.Tensor: final timestep [dim]
        """
        x = x_init
        for idx in range(len(steps) - 1):
            t = steps[:, idx]
            dt = steps[:, idx + 1] - steps[:, idx]
            x = self.solver.step(x, t, dt, **kwargs)
        return x
        
        
class SDE(ODE):
    """
    Abstract class for basic time dependent SDE of the form dx_t = f(x_t, t)dt + g(x_t, t) dB_t where (B_t) is some Brownian Motion. 
    We assume that f and g are given on [0,T].
    """
    @abstractmethod
    def diffusion(self, x_t : torch.Tensor,t : torch.Tensor) -> torch.Tensor:
        """
        Returns g(x_t,t)
        Args:
            x_t (torch.Tensor): input at time t [batch_size, dim]
            t (torch.Tensor): time t

        Returns:
            torch.Tensor: gives difussion part of SDE [batch_size, dim]
        """
        pass


