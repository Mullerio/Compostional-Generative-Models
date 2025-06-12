from .base_models import *
import random
from scipy.stats import norm
import numpy as np

"""Basic MLP where we embed the guidance parameter to some higher dimension for better learning"""

class CenterGuidanceTrainerOLD(BasicTrainer):
    def __init__(self, path: ConditionalProbabilityPath, model, p_uncond: float, 
                 optimizer=torch.optim.Adam, lr=0.01, model_type="FM", num_conditions=None, null_idx=0):
        super().__init__(model, optimizer, lr)
        assert model_type in ["FM", "Flow Matching", "Diff", "Diffusion"], "Model type not implemented"
        self.path = path
        self.p_uncond = p_uncond
        self.type = model_type
        self.num_conditions = num_conditions
        self.null_idx = null_idx  # index used for unconditional 

    def get_loss(self, n):
        z = self.path.p_data.sample(n)  # target data
        t = torch.rand(n, 1).to(z)

        # Sample conditioning indices
        y_index = torch.randint(0, self.num_conditions,(n,), device=z.device)  
        
        # Sample conditional path from centers
        cond_pt = self.path.sample_conditional_path(z, t)

        # Apply classifier-free guidance dropout
        drop_mask = (torch.rand(n, device=z.device) < self.p_uncond)  
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

                        if self.centers is not None:
                            nearest_idx = nearest_center_idx[i].item()
                            overlap_prob = self.center_rectangle_overlap_probs[nearest_idx, j].item()
                            weights.append(overlap_prob)
                        else:
                            weights.append(1.0)  

            if not valid_indices:
                if self.centers is not None:
                    fallback_idx = nearest_center_idx[i].item()
                    final_indices.append(torch.tensor(fallback_idx, device=device))
                    continue
                else:
                    raise ValueError(f"Point {xi} did not fall in any valid rectangle and no centers given.")

            # Normalize weights and sample
            weight_tensor = torch.tensor(weights, device=device, dtype=torch.float)
            norm_weights = weight_tensor / weight_tensor.sum()
            chosen = torch.multinomial(norm_weights, 1).item()
            final_indices.append(torch.tensor(valid_indices[chosen], device=device))

        return torch.stack(final_indices)
    
    
    

class RecGMGuidanceTrainer(GeneralGuidanceTrainer):
    def __init__(
        self,
        path: ConditionalProbabilityPath,
        model,
        p_uncond: float,
        num_conditions: int,
        optimizer=torch.optim.Adam,
        lr=0.01,
        model_type: str = "FM",
        std: list[float] = None,
        centers: list[list[torch.Tensor]] = None,  
        rectangle_boundaries: list[list[tuple[float, float]]] = None
    ):
        """
        Rectangle and Gaussian Mixture Guidance Trainer 

        Args:
            path (ConditionalProbabilityPath): _description_
            model (_type_): _description_
            p_uncond (float): _description_
            num_conditions (int): _description_
            optimizer (_type_, optional): _description_. Defaults to torch.optim.Adam.
            lr (float, optional): _description_. Defaults to 0.01.
            model_type (str, optional): _description_. Defaults to "FM".
            std (list[float], optional): _description_. Defaults to None.
            centers (list[list[torch.Tensor]], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__(path, model, p_uncond, num_conditions, optimizer, lr, model_type)

        if centers is None and rectangle_boundaries is None:
            raise ValueError("Give either some centers or rectangle boundaries, received both as None")

        self.centers = centers  # one list of centers per Gaussian Mixture
        self.rectangle_boundaries = rectangle_boundaries

        if std is None and centers is not None:
            self.std = [1.0] * len(centers)
        else:
            self.std = std

        if centers is not None and rectangle_boundaries is not None:
            self.center_rectangle_overlap_probs = torch.zeros((len(centers), len(rectangle_boundaries)))
            for i, modes in enumerate(centers):  # for each mixture, we want to compute the overlap between that mixture with each rectangle
                sig = self.std[i]
                for j, rect in enumerate(rectangle_boundaries):
                    probs = [rect_gauss_overlap_mass(mode, sig, rect) for mode in modes]
                    self.center_rectangle_overlap_probs[i, j] = 1 - np.prod([1 - p for p in probs]) # could also try something like max, or other weighting  

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Classification of x to closest center up to deviation or rectangle of which x is inside of. 
        This implementation is somewhat janky, TODO: improve readability and refactor, since in results it seems to not be balanced too well
        """
        batch_size = x.size(0)
        device = x.device
        final_indices = []

        # Precompute distances to all modes in all mixtures
        mode_tensors = []
        mixture_indices = []

        if self.centers is not None:
            for i, mode_list in enumerate(self.centers):
                for mode in mode_list:
                    mode_tensors.append(mode.to(device))
                    mixture_indices.append(i)
            all_modes = torch.stack(mode_tensors)  # [total_modes, D]
            dists = torch.cdist(x, all_modes)      # [batch_size, total_modes]
            closest_mode_indices = torch.argmin(dists, dim=1)
            closest_mixture_indices = torch.tensor([mixture_indices[idx] for idx in closest_mode_indices.tolist()], device=device)

        for i in range(batch_size): # very expensive, but dont know how to make this better without having a pre-learned classification model
            xi = x[i]
            valid_indices = []
            weights = []

            if self.centers is not None:
                nearest_mix = closest_mixture_indices[i].item()
                # Find distance to nearest mode of the mixture
                mode_list = self.centers[nearest_mix]
                dists_to_modes = [torch.norm(xi - m.to(device)) for m in mode_list]
                if min(dists_to_modes) <= self.std[nearest_mix]:
                    valid_indices.append(nearest_mix)
                    weights.append(1.0)

            if self.rectangle_boundaries is not None:
                for j, rect in enumerate(self.rectangle_boundaries):
                    lowers = torch.tensor([r[0] for r in rect], device=device)
                    uppers = torch.tensor([r[1] for r in rect], device=device)
                    in_rect = ((xi >= lowers) & (xi <= uppers)).all()
                    if in_rect:
                        idx = len(self.centers) + j if self.centers is not None else j
                        valid_indices.append(idx)
                        if self.centers is not None:
                            overlap_prob = self.center_rectangle_overlap_probs[closest_mixture_indices[i], j].item()
                            weights.append(overlap_prob)
                        else:
                            weights.append(1.0)

            if not valid_indices:
                if self.centers is not None:
                    fallback_idx = closest_mixture_indices[i].item()
                    final_indices.append(torch.tensor(fallback_idx, device=device))
                    continue
                else:
                    raise ValueError(f"Point {xi} is not inside any rectangle and no centers given.")

            weight_tensor = torch.tensor(weights, device=device, dtype=torch.float)
            norm_weights = weight_tensor / weight_tensor.sum()
            chosen = torch.multinomial(norm_weights, 1).item()
            final_indices.append(torch.tensor(valid_indices[chosen], device=device))

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