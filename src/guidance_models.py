from .base_models import *
import random
from scipy.stats import norm
import numpy as np

"""Basic MLP where we embed the guidance parameter to some higher dimension for better learning"""

class EmbeddedBasicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], conditional: bool = False, 
                 num_conditions: int = None, embedding_dim: int = None):
        """
        Args:
            input_dim: dimension of x
            hidden_dims: list of hidden layer sizes
            conditional: whether to use conditional input
            num_conditions: number of discrete conditions (clusters)
            embedding_dim: size of the embedding vectors (can be used for guidance)
        """
        super().__init__()
        self.conditional = conditional

        if conditional:
            assert num_conditions is not None and embedding_dim is not None, "For conditional, provide num_conditions and embedding_dim"
            self.embed = nn.Embedding(num_conditions, embedding_dim)
            cond_dim = embedding_dim
        else:
            cond_dim = 0

        layers = []
        current_dim = input_dim + 1 + cond_dim  
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, h))
            if i < len(hidden_dims) - 1:
                layers.append(nn.SiLU())
            current_dim = h
        layers.append(nn.Linear(current_dim, input_dim)) 

        self.mlp = nn.Sequential(*layers)    

    def forward(self, x: torch.Tensor, t: torch.Tensor, y_index: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim]
            t: [batch, 1]
            y_index: [batch] (long) -- index of conditioning cluster 
        """
        if self.conditional:
            if y_index is None:
                raise ValueError("Conditional model requires y_index")
            if y_index.dim() == 2:  # make sure y_index is shape [B], not [B,1]
                y_index = y_index.squeeze(-1)
            y_embed = self.embed(y_index)  # [batch, embedding_dim]
            input_tensor = torch.cat([x, t, y_embed], dim=-1)
        else:
            input_tensor = torch.cat([x, t], dim=-1)

        return self.mlp(input_tensor)     


class CenterGuidanceTrainerOLD(BasicTrainer):
    def __init__(self, path: ConditionalProbabilityPath, model, p_uncond: float, 
                 optimizer=torch.optim.Adam, lr=0.01, model_type="FM", num_conditions=None, null_idx=0):
        super().__init__(model, optimizer, lr)
        assert model_type in ["FM", "Flow Matching", "Diff", "Diffusion"], "Model type not implemented"
        self.path = path
        self.p_uncond = p_uncond
        self.type = model_type
        self.num_conditions = num_conditions
        self.null_idx = null_idx  # index used for unconditional pass

    def get_loss(self, n):
        z = self.path.p_data.sample(n)  # target data
        t = torch.rand(n, 1).to(z)

        # Sample conditioning indices (as integers)
        y_index = torch.randint(0, self.num_conditions, (n,), device=z.device)  # [n]
        
        # Sample conditional path from centers
        cond_pt = self.path.sample_conditional_path(z, t)

        # Apply classifier-free guidance dropout
        drop_mask = (torch.rand(n, device=z.device) < self.p_uncond)  # shape [n]
        y_masked = y_index.clone()
        y_masked[drop_mask] = self.null_idx  # set to "unconditional" index

        # Forward pass with masked y
        pred = self.model(cond_pt, t, y_masked)

        # Target vector field
        if self.type in ["FM", "Flow Matching"]:
            target = self.path.conditional_vector_field(cond_pt, z, t)
        elif self.type in ["Diff", "Diffusion"]:
            target = self.path.conditional_score(cond_pt, z, t)
        else:
            raise ValueError("Model type not implemented")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        return F.mse_loss(pred, target)

class GuidedVectorFieldOLD(ODE):
    def __init__(self, model: BasicMLP, guidance_scale: float, null_index: int):
        super().__init__()
        self.model = model
        self.guidance_scale = guidance_scale
        self.null_index = null_index

    def drift(self, x_t: torch.Tensor, t: torch.Tensor, y_index: torch.Tensor) -> torch.Tensor:
        # y_index: [batch] of integers
        uncond = self.model(x_t, t, torch.full_like(y_index, self.null_index))  # unconditional
        cond = self.model(x_t, t, y_index)  # conditional
        
        guidance_scale = self.guidance_scale

        return (1 - guidance_scale) * uncond + guidance_scale * cond



"""General guidance trainer, only need to implement a way to classify points to each label"""

class GeneralGuidanceTrainer(BasicTrainer):
    def __init__(self, path : ConditionalProbabilityPath, model,p_uncond : float, num_conditions : int, optimizer = torch.optim.Adam, lr = 0.01, model_type : str = "FM"):
        super().__init__(model, optimizer, lr)
        assert model_type == "FM" or model_type == "Flow Matching" or model_type == "Diff" or model_type == "Diffusion", "Model type not implemented"
        self.path = path
        self.p_uncond = p_uncond
        self.type = model_type
        self.labels_amount = num_conditions
        self.null_index = num_conditions - 1
   
    def set_null_index(self, index : int): 
        """
        We assume that the highest index is the null index, setter if other null indices are needed
        """
        self.null_index = index
    
    def get_loss(self, n):
        z = self.path.p_data.sample(n)  # Data to generate
        t = torch.rand(n, 1).to(z)
        y = self.classify(z)
        
        cond_pt = self.path.sample_conditional_path(z, t)

        # Classifier-free guidance: randomly drop conditioning, i.e set to null_index
        
        drop_mask = (torch.rand(n, 1, device=z.device) < self.p_uncond).squeeze(-1)  
        y_masked = y.clone()
        y_masked[drop_mask] = self.null_index
        
        
        pred = self.model(cond_pt, t, y_masked)

        if self.type == "FM" or self.type == "Flow Matching":
            target = self.path.conditional_vector_field(cond_pt,z,t)
        elif self.type == "Diff" or self.type == "Diffusion":
            target = self.path.conditional_score(cond_pt,z,t)
        else:
            raise ValueError("Model type not implemented")

        return F.mse_loss(pred, target)     
    
    @abstractmethod
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assigns each point in x to the index of its closest center.

        Args:
            x: [batch_size, data_dim]

        Returns:
            indices: [batch_size] long tensor of center indices
        """
        #dists = torch.cdist(x, torch.stack(self.centers))  # [batch_size, num_centers]
        #return torch.argmin(dists, dim=1)
        pass



"""Guidance, for when we know the means or approximate them as cluster centers from data, not using embeddings but learning the null center, does not work too well"""

class CenterGuidanceTrainer(GeneralGuidanceTrainer):
    def __init__(self, path : ConditionalProbabilityPath, model,p_uncond : float, num_conditions : int, centers : list[torch.Tensor], optimizer = torch.optim.Adam, lr = 0.01,model_type : str = "FM"):
        super().__init__(path, model, p_uncond, num_conditions, optimizer, lr, model_type)
        if centers is None: 
            raise ValueError("Give a valid list of centers")
        self.centers = centers

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assigns each point in x to the index of its closest center.

        Args:
            x: [batch_size, data_dim]

        Returns:
            indices: [batch_size] long tensor of center indices
        """
        dists = torch.cdist(x, torch.stack(self.centers))  # [batch_size, num_centers]
        return torch.argmin(dists, dim=1)


class RecCenterGuidanceTrainer(GeneralGuidanceTrainer):
    def __init__(self, path : ConditionalProbabilityPath, model,p_uncond : float, num_conditions : int, optimizer = torch.optim.Adam, lr = 0.01,model_type : str = "FM", std : list[float] = None, centers : list[torch.Tensor] = None,  rectangle_boundaries : list[list[tuple[float, float]]] = None):
        super().__init__(path, model, p_uncond, num_conditions, optimizer, lr, model_type)
        if centers is None and rectangle_boundaries is None: 
            raise ValueError("Give either some centers or rectangle boundaries, received both as None")
        self.centers = centers
        self.rectangle_boundaries = rectangle_boundaries
        if std is None and self.centers is not None:
            self.std = [1] * len(centers)
        else: 
            self.std = std
        
        if centers is not None: 
            self.center_rectangle_overlap_probs = torch.zeros((len(self.centers), len(self.rectangle_boundaries)))
            for i, center in enumerate(self.centers):
                for j, rect in enumerate(self.rectangle_boundaries):
                    self.center_rectangle_overlap_probs[i, j] = rect_gauss_overlap_mass(center, self.std[i], rect)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        final_indices = []

        # Precompute center distances if centers exist
        if self.centers is not None:
            center_tensor = torch.stack(self.centers).to(device)
            dists = torch.cdist(x, center_tensor)
            nearest_center_idx = torch.argmin(dists, dim=1)

        for i in range(batch_size):
            xi = x[i]
            valid_indices = []
            weights = []

            if self.centers is not None:
                nearest_idx = nearest_center_idx[i].item()
                dist = torch.norm(xi - self.centers[nearest_idx].to(device))
                if dist <= self.std[nearest_idx]:
                    valid_indices.append(nearest_idx)
                    weights.append(1.0)  # temporary weight (will normalize later)

            if self.rectangle_boundaries is not None:
                for j, rect in enumerate(self.rectangle_boundaries):
                    lowers = torch.tensor([r[0] for r in rect], device=device)
                    uppers = torch.tensor([r[1] for r in rect], device=device)
                    in_rectangle = ((xi >= lowers) & (xi <= uppers)).all()

                    if in_rectangle:
                        rect_idx = len(self.centers) + j if self.centers is not None else j
                        valid_indices.append(rect_idx)

                        # Weight based on overlap with nearest center (if applicable)
                        if self.centers is not None:
                            nearest_idx = nearest_center_idx[i].item()
                            overlap_prob = self.center_rectangle_overlap_probs[nearest_idx, j].item()
                            weights.append(overlap_prob)
                        else:
                            weights.append(1.0)  # equal weight for rectangles only

            if not valid_indices:
                # Fallback to nearest center (even outside std) if available
                if self.centers is not None:
                    fallback_idx = nearest_center_idx[i].item()
                    final_indices.append(torch.tensor(fallback_idx, device=device))
                    continue
                else:
                    raise ValueError(f"Point {xi} did not fall in any valid rectangle and no centers provided.")

            # Normalize weights and sample
            weight_tensor = torch.tensor(weights, device=device, dtype=torch.float)
            norm_weights = weight_tensor / weight_tensor.sum()
            chosen = torch.multinomial(norm_weights, 1).item()
            final_indices.append(torch.tensor(valid_indices[chosen], device=device))

        return torch.stack(final_indices)


    
    def classify2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assigns each point in x to one of its valid center/rectangle indices.
        If multiple are valid (inside rectangle and closest center), one is chosen at random uniformly. This is pretty slow!1!

        Args:
            x: [batch_size, data_dim]

        Returns:
            indices: [batch_size] long tensor of selected indices
        """
        batch_size = x.size(0)
        device = x.device

        dists = torch.cdist(x, torch.stack(self.centers))  # [batch_size, num_centers]
        nearest_center_idx = torch.argmin(dists, dim=1)  # [batch_size]

        rectangle_idxs = []
        for idx, rectangle in enumerate(self.rectangle_boundaries):
            lowers = torch.tensor([rectangle[i][0] for i in range(len(rectangle))], device=device)  # upper bounds tensor
            uppers = torch.tensor([rectangle[i][1] for i in range(len(rectangle))], device=device)  # upper bounds tensor
            
            in_rectangle = ((x >= lowers) & (x <= uppers)).all(dim=1)  # [batch_size] bool
            rect_idx = len(self.centers) + idx  # rectangle indices come AFTER centers

            rectangle_idxs.append((in_rectangle, rect_idx))

        final_indices = []
        for i in range(batch_size):
            valid = []
            
            nearest_idx = nearest_center_idx[i].item()
            dist_to_center = torch.norm(x[i] - self.centers[nearest_idx].to(device))
            if dist_to_center <= self.std[nearest_idx]:
                valid.append(nearest_idx)
            
            if self.rectangle_boundaries is not None:
                for in_rectangle, rect_idx in rectangle_idxs:
                    if in_rectangle[i]:
                        valid.append(rect_idx)
            
            if not valid:
                valid.append(nearest_center_idx[i].item())
            
            
            #SELECT INDICIES BASED ON WEIGHT OF MASS THAT OVERLAPS?!
            selected_idx = torch.tensor(valid[torch.randint(len(valid), (1,)).item()], device=device)
            final_indices.append(selected_idx)

        return torch.stack(final_indices)
    
    
class GuidedVectorField(ODE):
    def __init__(self, model: nn.Module, guidance_scale: float, null_index: int):
        super().__init__()
        self.model = model
        self.guidance_scale = guidance_scale
        self.null_index = null_index

    def drift(self, x_t: torch.Tensor, t: torch.Tensor, y_index: torch.Tensor):
        null_indices = torch.full_like(y_index, self.null_index)

        return ((1 - self.guidance_scale) * self.model(x_t, t, null_indices) + self.guidance_scale * self.model(x_t, t, y_index))
    


"""Basic Diffusion Guidance"""
class GuidedLangevin_withSchedule(SDE):
    def __init__(self, score_model : BasicMLP, alpha : Alpha, beta : Beta, sigma : float, guidance_scale : float, null_index : int,model_type = "score"):
        super().__init__()
        self.score = score_model
        self.beta = beta
        self.alpha = alpha
        self.sigma = sigma
        self.type = model_type
        self.guidance_scale = guidance_scale
        self.null_index = null_index


    def drift(self, x_t : torch.Tensor, t : torch.Tensor, y_index : torch.Tensor) -> torch.Tensor:
        null_indices = torch.full_like(y_index, self.null_index)

        score = ((1 - self.guidance_scale) * self.score(x_t, t, null_indices) + self.guidance_scale * self.score(x_t, t, y_index))
        beta_t = self.beta(t)
        if self.type == "noise":
            score = score/-beta_t
        beta_dt = self.beta.dt(t)
        alpha_t = self.alpha(t) 
        alpha_dt = self.alpha.dt(t)

        return  (beta_t**2 * alpha_dt/alpha_t - beta_dt* beta_t+ self.sigma**2/2 ) * score+ alpha_dt/alpha_t *x_t
    
    def diffusion(self, x_t : torch.Tensor, t : torch.Tensor, y_index : torch.Tensor) -> torch.Tensor:
        
        return self.sigma * torch.randn_like(x_t)
    

"""General Guidance trainer, does not work too well without better *guidance*"""
class GuidanceTrainer(BasicTrainer):
    def __init__(self, path : ConditionalProbabilityPath, model,p_uncond : float, optimizer = torch.optim.Adam, lr = 0.01, model_type : str = "FM"):
        super().__init__(model, optimizer, lr)
        assert model_type == "FM" or model_type == "Flow Matching" or model_type == "Diff" or model_type == "Diffusion", "Model type not implemented"
        self.path = path
        self.p_uncond = p_uncond
        self.type = model_type
    
    def get_loss(self, n):
        z = self.path.p_data.sample(n)  # Data to generate
        y = self.path.p_data.sample(n)  # Use another datapoint as the "guidance"

        t = torch.rand(n, 1).to(z)
        cond_pt = self.path.sample_conditional_path(z, t)

        # Classifier-free guidance: randomly drop conditioning
        drop_mask = (torch.rand(n, 1, device=cond_pt.device) < self.p_uncond)
        y_masked = y.clone()
        y_masked[drop_mask.squeeze()] = 0.0

        pred = self.model(cond_pt, t, y_masked)

        if self.type == "FM" or self.type == "Flow Matching":
            target = self.path.conditional_vector_field(cond_pt,z,t)
        elif self.type == "Diff" or self.type == "Diffusion":
            target = self.path.conditional_score(cond_pt,z,t)

        return F.mse_loss(pred, target)


    def forward(self, x: torch.Tensor, t: torch.Tensor, y_index: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim]
            t: [batch, 1]
            y_index: [batch] (long) -- index of conditioning cluster
        """
        if self.conditional:
            if y_index is None:
                raise ValueError("Conditional model requires y_index")
            y_embed = self.embed(y_index)  # [batch, embedding_dim]
            input_tensor = torch.cat([x, t, y_embed], dim=-1)
        else:
            input_tensor = torch.cat([x, t], dim=-1)

        return self.mlp(input_tensor)