from geomloss import SamplesLoss
import torch


def print_distribution_metrics(generated, ground_truth):
    """
    Prints the 2-Wasserstein distance, Energy MMD and Gaussian MMD between two sets of samples from two distributions. 
    """
    
    lossfn_1 = SamplesLoss("sinkhorn", blur=0.1, p = 2)
    lossfn_2 = SamplesLoss("energy", blur=0.1)
    lossfn_3 = SamplesLoss("gaussian", blur=0.1)
    
    print("=== Distances between Samples ===")
    print(f"\n Wasserstein-2 Distance: {lossfn_1(generated, ground_truth).item()}")
    print(f"\n Energy MMD: {lossfn_2(generated, ground_truth).item()}")
    
    print(f"\n Gaussian MMD: {lossfn_3(generated, ground_truth).item()}")
    
    print("=== Coverge Metrics ===")
    precision, recall = compute_precision_recall(generated, ground_truth)
    
    print(f"\n Precision, i.e. the fraction of generated Samples that fall into the neighborhood of any ground truth sample: {precision}")
    print(f"\n Recall, i.e. the fraction of real samples that fall into the neighborhood of any generated sample: {recall}")

    

def compute_precision_recall(generated, ground_truth, k=3):
    """
    real_samples: (N, D) torch tensor
    gen_samples:  (M, D) torch tensor
    k: number of neighbors to use to define radius (e.g. k=3)
    
    Returns:
        precision, recall: fractions in [0, 1]
    """
    dists_real = torch.cdist(ground_truth, ground_truth)  
    radii_real = torch.sort(dists_real, dim=1)[0][:, k]  # distance to kth neighbor for each ground truth sample

    dists_gen_to_real = torch.cdist(generated, ground_truth) 
    in_ball = dists_gen_to_real <= radii_real.unsqueeze(0)  
    precision = (in_ball.any(dim=1).float().mean()).item()

    dists_real_to_gen = torch.cdist(ground_truth, generated) 
    in_ball_recall = dists_real_to_gen <= radii_real.unsqueeze(1)  
    recall = (in_ball_recall.any(dim=1).float().mean()).item()

    return precision, recall