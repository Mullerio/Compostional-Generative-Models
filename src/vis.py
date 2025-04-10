import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_samples(samples: torch.Tensor, 
                 title: str = "Sampled Distribution", 
                 figsize: tuple = (6, 6), 
                 point_alpha: float = 0.3, 
                 point_size: int = 8, 
                 kde: bool = True,
                 contour: bool = True,
                 cmap: str = "mako",
                 save_path: str = None):
    """
    Visualize 2D distribution samples with optional KDE and contours.

    Args:
        samples (torch.Tensor): Tensor of shape (N, 2)
        title (str): Plot title
        figsize (tuple): Size of the figure
        point_alpha (float): Transparency for scatter points
        point_size (int): Size of scatter points
        kde (bool): Whether to plot kernel density estimate
        contour (bool): Whether to overlay contour lines
        cmap (str): Colormap for KDE
        save_path (str, optional): Path to save the plot
    """
    samples_np = samples.detach().cpu().numpy()
    x, y = samples_np[:, 0], samples_np[:, 1]

    plt.figure(figsize=figsize)
    plt.style.use("dark_background")
    sns.set_theme(style="dark", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    if kde:
        sns.kdeplot(
            x=x, y=y,
            fill=True,
            cmap=cmap,
            bw_adjust=0.3,
            #levels=100,
            thresh=0.00,
            alpha=0.5,
        )

    if contour:
        sns.kdeplot(
            x=x, y=y,
            cmap="gray",
            bw_adjust=0.5,
            levels=10,
            linewidths=1
        )

    #plt.scatter(x, y, color='black', s=point_size, alpha=point_alpha, edgecolors='none')
    plt.title(title, fontsize=14)
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()