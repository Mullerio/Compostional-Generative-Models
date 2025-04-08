from .path_lib import *
import torch.nn as nn
import torch
import torch.nn.functional as F

class BasicMLP(nn.Module):
    """
    Basic MLP used to approximate the target vector field in flow models.
    """
    def __init__(self, input_dim : int, hidden_dims : list[int]):
        """
        Args:
            input_dim (int): input dim of nn
            hidden_dims (list[int]): list of hidden dimensions, including output dim
        """
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("Hidden dims list must be non-empty")
        self.mlp = nn.Sequential(nn.Linear(input_dim+1, hidden_dims[0]))
        
        if len(hidden_dims) - 2 > 0:
            self.mlp.add_module("0Activation",nn.SiLU())
            
        for i in range(len(hidden_dims)-1):
            self.mlp.add_module(f"Layer{i+1}",nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if len(hidden_dims) - i - 2> 0:
                self.mlp.add_module(f"Activation{i+1}",nn.SiLU())
        
        self.mlp.add_module("OutLayer", nn.Linear(hidden_dims[-1],input_dim))
        
    def forward(self, x : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of layer to appprox vector field

        Args:
            x (torch.Tensor): [batch_size, dim]
            t (torch.Tensor): [batch_size, 1]

        Returns:
            torch.Tensor: approx of vectorfield
        """
        return self.mlp(torch.concat([x,t],dim=-1))
    

class VectorFieldODE(ODE):
    def __init__(self, mlp : BasicMLP):
        super().__init__()
        self.mlp = mlp
    
    def drift(self, x_t : torch.Tensor, t : torch.Tensor):
        return self.mlp(x_t,t)

class Langevin_withFLow(SDE):
    def __init__(self, eps : float, flow : BasicMLP, score : BasicMLP):
        super().__init__()
        self.eps = eps
        self.flow = flow
        self.score = score
    
    def drift(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        Drift coefficient of Langevin with a Flow model.

        Args:
            x_t (torch.Tensor): [batch_size, dim]
            t (torch.Tensor): [batch_size]
        """
        vf_t = self.flow(x_t,t)
        score_t = self.score(x_t,t)
        
        return self.eps ** 2/2 * score_t + vf_t
    
    def diffusion(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        
        return torch.randn_like(x_t) * self.eps 
        
class Langevin_Schedule(SDE):
    def __init__(self, score_model : BasicMLP, schedule : Beta):
        super().__init__()
        self.score = score_model
        self.reverse_schedule = schedule

    def drift(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        
        score = self.score(x_t,t)
        schedule = self.reverse_schedule(t)
        
        return schedule ** 2/2 * score
    
    def diffusion(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        
        return self.reverse_schedule(t) * torch.randn_like(x_t)
    
    
class BasicTrainer(ABC):
    def __init__(self, model, optimizer : torch.optim.Optimizer = torch.optim.Adam, lr : float = 0.01):
        super().__init__()
        self.model = model
        self.optim = optimizer
        self.lr = lr

    @abstractmethod
    def get_loss(self, **kwargs):
        """
        Lossfunction for Training

        Args:
            possible args depending on modekl
        """
        pass

    def set_optimizer(self, optimizer : torch.optim.Optimizer):
        """
        Setter for the optimizer 
        """
        self.optim = optimizer

    def train_loop(self, device : torch.device, num_epochs : int, lr : float = None, **kwargs):
        """
        Basic Train loop for the model, using the optimizer of the class 

        Args:
            device (torch.device): _description_
            num_epochs (int): _description_
            lr (float, optional): _description_. Defaults to None.
        """
        self.model = self.model.to(device)
        self.model.train()
        self.lr = lr or self.lr  # use given lr, otherwise get new lr
        optimizer = self.optim(self.model.parameters(), lr=self.lr)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad() 
            loss = self.get_loss(**kwargs)
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print(f"In Epoch {epoch} we have a loss of {loss}")
            
        self.model.eval()
            
class FlowDiffTrainer(BasicTrainer):
    def __init__(self, path : ConditionalProbabilityPath, modeltype : str, model, optimizer = torch.optim.Adam, lr = 0.01):
        super().__init__(model, optimizer, lr)
        self.path = path
        self.modeltype = modeltype
        
    def get_loss(self, n : int):
        """Standard conditional flow matching loss for n datapoints i.e. approx of E[mlp - condvf]"""
        z = self.path.p_data.sample(n) #[n, dim]
        t = torch.rand(n, 1).to(z) #[n, 1]
        cond_pt = self.path.sample_conditional_path(z, t) #[n, dim]
        
        u_theta = self.model(cond_pt, t)
        if self.modeltype == "FM" or self.modeltype == "FlowMatching":
            label = self.path.conditional_vector_field(cond_pt, z, t)   
        elif self.modeltype == "Diff" or self.modeltype == "Diffusion":
            label = self.path.conditional_score(cond_pt,z,t)
        else:
            raise ValueError("Type not Supported, either FlowMatching or Diffusion")
        
        return F.mse_loss(u_theta,label)
        
    
    

