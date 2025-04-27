from .path_lib import *
import torch.nn as nn
import torch
import torch.nn.functional as F

class BasicMLP(nn.Module):
    """
    Basic MLP used to approximate the target vector field in flow models.
    """
    def __init__(self, input_dim : int, hidden_dims : list[int], conditional : bool = False, conditional_dim : int = 2):
        """
        Args:
            input_dim (int): input dim of nn
            hidden_dims (list[int]): list of hidden dimensions, including output dim
        """
        super().__init__()
        self.conditional = conditional
        
        if len(hidden_dims) == 0:
            raise ValueError("Hidden dims list must be non-empty")
        
        extra = 0
        if conditional:
            extra = conditional_dim
            
        layers = []
        current_dim = input_dim + 1 + extra  # x + t + cond_embed
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, h))
            if i < len(hidden_dims) - 1:
                layers.append(nn.SiLU())
            current_dim = h
        layers.append(nn.Linear(current_dim, input_dim))  # output vector field

        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x : torch.Tensor, t : torch.Tensor, y : torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of layer to appprox vector field

        Args:
            x (torch.Tensor): [batch_size, dim]
            t (torch.Tensor): [batch_size, 1]

        Returns:
            torch.Tensor: approx of vectorfield
        """
        
        if self.conditional:
            if y is not None:
                return self.mlp(torch.concat([x,t,y],dim=-1))
            else:
                raise ValueError("For a conditional MLP provide a conditional")
        return self.mlp(torch.concat([x,t],dim=-1))
    

class VectorFieldODE(ODE):
    """Wrapper to get ODE out of MLP"""
    def __init__(self, mlp : BasicMLP):
        super().__init__()
        self.mlp = mlp
    
    def drift(self, x_t : torch.Tensor, t : torch.Tensor):
        return self.mlp(x_t,t)

class Langevin_withFLow(SDE):
    """
    General Langevin SDE given a score and flow model 
    """
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
    """
    For Gaussian paths, the SDE with the flow, given a score approximation, can be derived by hand 
    """
    def __init__(self, score_model : BasicMLP, alpha : Alpha, beta : Beta, sigma : float, model_type = "score"):
        super().__init__()
        self.score = score_model
        self.beta = beta
        self.alpha = alpha
        self.sigma = sigma
        self.type = model_type

    def drift(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        score = self.score(x_t,t)
        beta_t = self.beta(t)
        if self.type == "noise":
            score = score/-beta_t
        beta_dt = self.beta.dt(t)
        alpha_t = self.alpha(t) 
        alpha_dt = self.alpha.dt(t)

        return  (beta_t**2 * alpha_dt/alpha_t - beta_dt* beta_t+ self.sigma**2/2 ) * score+ alpha_dt/alpha_t *x_t
    
    def diffusion(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        
        return self.sigma * torch.randn_like(x_t)
    
    

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
            possible args depending on model
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
            

class ScoreFromVectorField(torch.nn.Module):
    """
    FOr Gaussian paths, approximation of score from given flow model/learned vector field
    """
    def __init__(self, vector_field: BasicMLP, alpha: Alpha, beta: Beta):
        super().__init__()
        self.vector_field = vector_field
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - score: (bs, dim)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        dt_alpha_t = self.alpha.dt(t)
        dt_beta_t = self.beta.dt(t)

        num = alpha_t * self.vector_field(x,t) - dt_alpha_t * x
        den = beta_t ** 2 * dt_alpha_t - alpha_t * dt_beta_t * beta_t

        return num / den     



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
        elif self.modeltype == "Diff" or self.modeltype == "Diffusion" or self.modeltype == "Score":
            label = self.path.conditional_score(cond_pt,z,t)
        else:
            raise ValueError("Type not Supported, either FlowMatching or Diffusion")
        
        return F.mse_loss(u_theta,label)
    
"""
Noise predictor in the case of gaussian probability paths, only varies from the above case in the gaussian case by constants, use "noise" when sampling using this
"""        
class NoisePredictorTrainer(BasicTrainer):
    def __init__(self, path : GaussianConditionalProbabilityPath, model, optimizer = torch.optim.Adam, lr = 0.01):
        super().__init__(model, optimizer, lr)
        self.path = path

    def get_loss(self, n : int):
        z = self.path.p_data.sample(n)
        t = torch.rand(n,1).to(z)
        alpha_t = self.path.alpha(t) 
        beta_t = self.path.beta(t)
        eps = torch.randn_like(z).to(z)
        x_t = alpha_t * z + beta_t * eps
        
        noise_pred = self.model(x_t,t)
        return F.mse_loss(noise_pred,eps)


