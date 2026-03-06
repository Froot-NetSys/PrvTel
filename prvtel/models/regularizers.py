import torch
from torch.distributions.normal import Normal

class LatentRegularizer:
    """Base class for latent space regularization strategies"""
    def compute_loss(self, mu_z, logsigma_z):
        raise NotImplementedError

class KLDivergenceRegularizer(LatentRegularizer):
    def __init__(self, device):
        self.device = device
    
    def compute_loss(self, mu_z, logsigma_z):        
        # Prior distribution p(z) = N(0,1)
        p = Normal(torch.zeros_like(mu_z), torch.ones_like(mu_z))
        # Approximate posterior q(z|x) = N(mu_z, sigma_z)
        q = Normal(mu_z, torch.exp(logsigma_z))
        return torch.sum(torch.distributions.kl_divergence(q, p))

class MMDRegularizer(LatentRegularizer):
    def __init__(self, device, sigmas=[1, 2, 4, 8, 16]):
        self.device = device
        self.sigmas = sigmas
    
    def compute_kernel(self, x, y, sigma=1.0):
        """
        Computes the RBF (Gaussian) kernel between two sets of vectors.
        This is used as part of Maximum Mean Discrepancy (MMD) calculation,
        which measures the distance between two probability distributions.
        
        The kernel trick allows us to implicitly compute similarities in a 
        high-dimensional feature space without explicitly computing the mapping.
        """
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/(2 * sigma * sigma)
        return torch.exp(-kernel_input)
        
    def compute_loss(self, mu_z, logsigma_z):
        """
        Computes Maximum Mean Discrepancy (MMD) between two samples x and y.
        MMD is used as a regularization term to ensure the encoded latent space
        follows a desired distribution (typically standard normal).
        
        MMD measures how different two distributions are by comparing their moments
        in a high-dimensional feature space (implicitly through the kernel trick).
        A lower MMD value indicates the distributions are more similar.
        """
        true_samples = torch.randn_like(mu_z, device=self.device)
        mmd = 0
        for sigma in self.sigmas:
            x_kernel = self.compute_kernel(mu_z, mu_z, sigma)
            y_kernel = self.compute_kernel(true_samples, true_samples, sigma)
            xy_kernel = self.compute_kernel(mu_z, true_samples, sigma)
            mmd += x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd / len(self.sigmas)
