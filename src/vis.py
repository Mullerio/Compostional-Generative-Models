import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.path_lib import *
from typing import Optional
from matplotlib import cm

def plot_samples(samples: torch.Tensor, 
                 title: str = "Sampled Distribution", 
                 point_alpha: float = 0.7, 
                 point_size: int = 8, 
                 kde: bool = False,
                 contour: bool = False,
                 scatter : bool = True,
                 grid : bool = True,
                 contour_cmap: str = "gray",
                 plot_range : tuple[tuple[float,float], tuple[float,float]] = None,
                 ax : Optional[plt.Axes] = None,
                 save_path: str = None):
    """
    Visualize 2D distribution samples with optional grid, contour etc.

    Args:
        samples (torch.Tensor): Tensor of shape (N, 2)
        title (str): Plot title
        figsize (tuple): Size of the figure
        point_alpha (float): Transparency for scatter points
        point_size (int): Size of scatter points
        contour (bool): Whether to overlay contour lines
        cmap (str): Colormap for KDE
        save_path (str, optional): Path to save the plot
    """
    samples_np = samples.detach().cpu().numpy()
    x, y = samples_np[:, 0], samples_np[:, 1]

    if ax is None: 
        ax = plt.gca()   
    
    #in case we want to overlay a kde 
    if kde: 
        sns.kdeplot(
        x=x, y=y,
        fill=True,
        cmap="mako", # styling to use
        levels=100, # amount of contours
        thresh=0.01, # at what level the conoutrs are drawn
        alpha = 0.1 # low transperacy to not overlay the scatter or contour
        )  

    if contour:
        sns.kdeplot(
            x=x, y=y,
            cmap=contour_cmap,
            bw_adjust=1,
            levels=10,
            linewidths=1,
            ax=ax)

    if scatter:
        ax.scatter(x, y, color='black', s=point_size, alpha=point_alpha, edgecolors='none')
    ax.set_title(title, fontsize=14)
    
    if grid:
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5) 
    
    if save_path:
        ax.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if plot_range is not None: 
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])    
    else: 
        ax.axis("equal")



def plot_trajectories(trajectories: torch.Tensor, 
                      title: str = "Sampled Trajectories", 
                      num_to_plot: int = 100,
                      alpha: float = 0.7,
                      linewidth: float = 1.0,
                      start_end_points: bool = True,
                      plot_range: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
                      ax: Optional[plt.Axes] = None,
                      save_path: str = None):
    """
    Plot 2D traj

    Args:
        trajectories: Tensor of shape (batch_size, num_steps, 2)
        num_to_plot: Number of individual trajectories to draw
        alpha: Transparency of the lines
        linewidth: Line thickness
        start_end_points: Whether to mark start and end with dots
        plot_range: ((xmin, xmax), (ymin, ymax))
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    data = trajectories.detach().cpu().numpy()
    for traj in data[:num_to_plot]:
        ax.plot(traj[:, 0], traj[:, 1], color='black', alpha=alpha, linewidth=linewidth)
        if start_end_points:
            ax.scatter(traj[0, 0], traj[0, 1], color='green', s=10)  # Start
            ax.scatter(traj[-1, 0], traj[-1, 1], color='red', s=10)  # End

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle='--', linewidth=0.5)

    if plot_range is not None:
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        ax.set_aspect("auto")
    else:
        ax.axis("equal")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')    
    
    
    
def plot_kde(samples: torch.Tensor, 
                 title: str = "Kernel Density Estimate of Samples", 
                 figsize: tuple = (6, 6), 
                 levels : int = 100,
                 thresh : float = 0.00, 
                 cmap: str = "mako",
                 ax : Optional[plt.Axes] = None,
                 save_path: str = None):
    
    
    samples_np = samples.detach().cpu().numpy()
    x, y = samples_np[:, 0], samples_np[:, 1]

    if ax is None: 
        ax = plt.gca()   
        
    cmap_array = cm.get_cmap(cmap)
    dark_color = cmap_array(0)  # cmap values go from 0 (darkest) to 1 (lightest)
    ax.set_facecolor(dark_color)   # extend backgroudn since not everything is filled for some reason?!

    sns.kdeplot(
    x=x, y=y,
    fill=True,
    cmap=cmap, # styling to use
    levels=levels, # amount of contours
    thresh=thresh, # at what level the conoutrs are drawn
    )  
    ax.set_title(title, fontsize=14)
    ax.axis("equal")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')



def make_grid(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    bins: int,
    x_offset: float = 0.0,
    device: Optional[torch.device] = None
):
    x = torch.linspace(x_bounds[0], x_bounds[1], bins)
    y = torch.linspace(y_bounds[0], y_bounds[1], bins)
    if device:
        x = x.to(device)
        y = y.to(device)
    x = x + x_offset
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    return X, Y, grid
    
    
def plot_logDensity(density : LogDensity, x_bounds : tuple[float, float], y_bounds : tuple[float, float], bins : int = 100, ax : Optional[plt.Axes] = None, offset : float = 0.0, device: Optional[torch.device] = None, **kwargs):
    ax = plt.gca() if ax is None else ax
    X, Y, xy = make_grid(x_bounds, y_bounds, bins, offset, device)
    Z = density.log_density(xy).reshape(bins, bins).T
    ax.imshow(Z.cpu(), extent=[*x_bounds, *y_bounds], origin='lower', **kwargs)
