import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm
import logging
import numpy as np
import math
import hashlib
import array
import statistics
import time
import pandas as pd

# Constants
BETA_SCALE = 0.01

def sigmoid(x):
    """Custom sigmoid function that scales the output to range [0, 0.01]"""
    return 1 / (1 + np.exp(-x)) * BETA_SCALE

def sketch_topk(cs, data, k):
    """Find the top-k most frequent elements in the data using Count-Sketch estimation
    
    Parameters
    ----------
    cs : CountSketch
        Count-Sketch data structure that maintains frequency estimates
    data : iterable
        Data elements to query frequencies for
    k : int
        Number of top elements to return
    
    Returns
    -------
    list
        k most frequent elements sorted by their estimated frequencies
        
    Notes
    -----
    This function uses the Count-Sketch data structure to estimate frequencies
    of elements and returns the k items with highest estimated frequencies.
    The Count-Sketch provides approximate frequency counts using sub-linear space.
    """
    # Query the frequency estimates for each item from the Count-Sketch
    ans = {}
    for item in data:
        freq = cs.query(item)
        ans[item] = freq
    
    # Sort items by their estimated frequencies in descending order
    # and return the top k elements
    return sorted(ans, key=ans.get, reverse=True)[:k]
    
class CountSketch:
    """Count-Sketch data structure for approximating frequencies of elements in a stream
    
    Parameters
    ----------
    width : int
        Number of columns (t) in the sketch matrix
    depth : int
        Number of rows (d) in the sketch matrix
    rho : float, optional
        Privacy parameter for differential privacy. If provided, adds Gaussian noise
        
    Attributes
    ----------
    width : int
        Width of the sketch matrix
    depth : int
        Depth of the sketch matrix
    gamma : float
        Scale parameter (1/width)
    beta : float
        Privacy parameter (1/e^depth)
    sketch_matrix : list
        List of arrays representing the sketch matrix
    sigma : float
        Standard deviation of Gaussian noise (only if rho provided)
    """
    
    def __init__(self, width, depth, rho=None):
        self.width = width
        self.depth = depth
        self.gamma = 1.0/width
        self.beta = 1.0/math.exp(depth)
        
        # Initialize sketch matrix
        self.sketch_matrix = self._initialize_matrix(rho)
        self.sigma = math.sqrt(math.log(2.0/self.beta)/rho) if rho else None

    def _initialize_matrix(self, rho):
        """Initialize the sketch matrix, optionally with noise
        
        Parameters
        ----------
        rho : float or None
            Privacy parameter. If provided, adds Gaussian noise
            
        Returns
        -------
        list
            List of arrays representing the sketch matrix
        """
        if not rho:
            return [array.array("f", [0.0] * self.width) for _ in range(self.depth)]
        
        sigma = math.sqrt(math.log(2.0/self.beta)/rho)
        return [
            array.array("f", np.random.normal(0, sigma, self.width))
            for _ in range(self.depth)
        ]

    def _hash_indices(self, x):
        """Generate hash indices for an item
        
        Parameters
        ----------
        x : hashable
            Item to hash
            
        Returns
        -------
        generator
            Yields indices for each row of the sketch matrix
        """
        md5 = hashlib.md5(str(hash(x)).encode('utf-8'))
        for i in range(self.depth):
            md5.update(str(i).encode('utf-8'))
            yield int(md5.hexdigest(), 16) % self.width

    def _hash_signs(self, x):
        """Generate hash signs (+1/-1) for an item
        
        Parameters
        ----------
        x : hashable
            Item to hash
            
        Returns
        -------
        generator
            Yields +1 or -1 for each row
        """
        sha = hashlib.sha256(str(hash(x)).encode('utf-8'))
        for i in range(self.depth):
            sha.update(str(i).encode('utf-8'))
            yield 1 if int(sha.hexdigest(), 16) % 2 else -1

    def update(self, x, value=1):
        """Update frequency count for an item
        
        Parameters
        ----------
        x : hashable
            Item to update
        value : int, default=1
            Value to add to the count
        """
        for table, i, sign in zip(
            self.sketch_matrix, 
            self._hash_indices(x), 
            self._hash_signs(x)
        ):
            table[i] += sign * value

    def query(self, x):
        """Query the estimated frequency of an item
        
        Parameters
        ----------
        x : hashable
            Item to query
            
        Returns
        -------
        float
            Estimated frequency of the item
        """
        estimates = [
            sign * table[i]
            for table, i, sign in zip(
                self.sketch_matrix,
                self._hash_indices(x),
                self._hash_signs(x)
            )
        ]
        return statistics.median(estimates)

class Encoder(nn.Module):
    """Encoder network that takes input data and outputs latent space parameters
    
    Parameters
    ----------
    input_dim : int
        Dimension of input data
    latent_dim : int
        Dimension of latent space
    hidden_dim : int, default=32
        Dimension of hidden layers
    activation : torch.nn.Module, default=nn.Tanh
        Activation function to use
    device : str, default="gpu"
        Device to run the model on
    """

    def __init__(
        self, input_dim, latent_dim, hidden_dim=32, activation=nn.Tanh, device="gpu",
    ):
        super().__init__()
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Encoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Encoder: {device} specified, {self.device} used")
        output_dim = 2 * latent_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        outs = self.net(x)
        mu_z = outs[:, : self.latent_dim]
        logsigma_z = outs[:, self.latent_dim :]
        return mu_z, logsigma_z

class Decoder(nn.Module):
    """Decoder network that takes latent vectors and outputs reconstructed data
    
    Parameters
    ----------
    latent_dim : int
        Dimension of latent space
    num_continuous : int
        Number of continuous variables
    num_categories : list, default=[0]
        List of number of categories for each categorical variable
    hidden_dim : int, default=32
        Dimension of hidden layers
    activation : torch.nn.Module, default=nn.Tanh
        Activation function to use
    device : str, default="gpu"
        Device to run the model on
    """

    def __init__(
        self,
        latent_dim,
        num_continuous,
        num_categories=[0],
        hidden_dim=32,
        activation=nn.Tanh,
        device="gpu",
    ):
        super().__init__()

        output_dim = num_continuous + sum(num_categories)
        self.num_continuous = num_continuous
        self.num_categories = num_categories

        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Decoder: {device} specified, {self.device} used")

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)

class GraphVAE(nn.Module):
    """VAE with graph prior knowledge incorporated into the loss function
    """
    
    def __init__(self, encoder, decoder, lr=1e-4, use_beta=False, graph_prior=None, alpha=0.0001):
        super().__init__()
        self.encoder = encoder.to(encoder.device)
        self.decoder = decoder.to(decoder.device)
        self.device = encoder.device
        self.num_categories = self.decoder.num_categories
        self.num_continuous = self.decoder.num_continuous
        self.use_beta = use_beta
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr = lr
        
        # Graph prior related parameters
        self.graph_prior = graph_prior
        if graph_prior is not None:
            self.graph_prior = torch.FloatTensor(graph_prior).to(self.device)
        self.alpha = alpha

    def forward(self, X):
        """
        Encodes and then decodes a batch of data points.
        """
        mu_z, logsigma_z = self.encoder(X)

        # Reparameterization trick
        s = torch.randn_like(mu_z)
        z_samples = mu_z + s * torch.exp(logsigma_z)

        # Decode the sampled latent vectors
        X_hat = self.decoder(z_samples)
        return X_hat, mu_z, logsigma_z

    def compute_latent_graph_loss(self, z):
        """
        Compute graph structure loss based on latent representations
        
        Parameters
        ----------
        z : torch.Tensor
            Latent space representations
            
        Returns
        -------
        torch.Tensor
            Graph structure loss value
        """
        if self.graph_prior is None:
            return torch.tensor(0.0).to(self.device)

        batch_size = z.shape[0]
        
        # If batch size is smaller than graph_prior size, use a subset of graph_prior
        if batch_size < self.graph_prior.shape[0]:
            graph_prior_batch = self.graph_prior[:batch_size, :batch_size]
        else:
            # If batch size is larger, tile the graph_prior
            repeats_needed = (batch_size + self.graph_prior.shape[0] - 1) // self.graph_prior.shape[0]
            graph_prior_tiled = self.graph_prior.repeat(repeats_needed, repeats_needed)
            graph_prior_batch = graph_prior_tiled[:batch_size, :batch_size]

        # Compute similarity matrix from latent representations
        z_norm = F.normalize(z, p=2, dim=1)
        S = torch.mm(z_norm, z_norm.t())
        
        # Scale similarity to [0,1] range
        S = (S + 1) / 2
        
        # Compute graph loss (Frobenius norm of difference)
        graph_loss = torch.norm(S - graph_prior_batch, p='fro')
        
        # # Normalize by batch size to make loss independent of batch size
        # graph_loss = graph_loss / (batch_size * batch_size)
        
        return graph_loss

    def compute_top_k_loss(self, X, X_hat, K=100, sample_size=1000):
        """
        Compute the loss based on the difference between top-K values of input and output data.
        
        Parameters
        ----------
        X : torch.Tensor
            Original input data
        X_hat : torch.Tensor
            Reconstructed output data
        K : int
            Number of top elements to consider
        sample_size : int
            Number of samples to use for estimating top-K values
            
        Returns
        -------
        torch.Tensor
            Top-K loss value
        """
        # Randomly sample indices
        start_time = time.time()
        indices = torch.randperm(X.size(0))[:min(sample_size, X.size(0))]
        # print(f"Sampling indices took {time.time() - start_time:.4f} seconds")

        # Sampled data
        start_time = time.time()
        X_sample = X[indices]
        X_hat_sample = X_hat[indices]
        # print(f"Sampling data took {time.time() - start_time:.4f} seconds")

        # Initialize CountSketch for each column
        start_time = time.time()
        cols = range(X_sample.size(1))  # Assuming columns are indexed by integers
        col2CS_input = {col: CountSketch(1000, 3, rho=2) for col in cols}
        col2CS_output = {col: CountSketch(1000, 3, rho=2) for col in cols}
        # print(f"Initializing CountSketch took {time.time() - start_time:.4f} seconds")

        # Vectorized updates for input data
        X_numpy = X_sample.detach().cpu().numpy()
        for col in cols:
            # Update all rows for this column at once
            for val in X_numpy[:, col]:
                col2CS_input[col].update(int(val))
                
        # Vectorized updates for output data
        X_hat_numpy = X_hat_sample.detach().cpu().numpy()
        for col in cols:
            # Update all rows for this column at once
            for val in X_hat_numpy[:, col]:
                col2CS_output[col].update(int(val))
                
        # Determine the range of values to consider for top-K
        start_time = time.time()
        min_val = int(torch.floor(torch.min(X_sample.min(), X_hat_sample.min())).item())
        max_val = int(torch.ceil(torch.max(X_sample.max(), X_hat_sample.max())).item())
        value_range = range(min_val, max_val + 1)
        # print(f"Determining value range took {time.time() - start_time:.4f} seconds")

        # Calculate top-K values
        start_time = time.time()
        top_k_input = [sketch_topk(col2CS_input[col], value_range, K) for col in cols]
        top_k_output = [sketch_topk(col2CS_output[col], value_range, K) for col in cols]
        print(top_k_input)
        # print(f"Calculating top-K values took {time.time() - start_time:.4f} seconds")

        # Compute the difference between top-K values
        start_time = time.time()
        top_k_loss = sum(
            len(set(top_k_input[col]) - set(top_k_output[col])) +
            len(set(top_k_output[col]) - set(top_k_input[col]))
            for col in cols
        )
        # print(f"Computing top-K loss took {time.time() - start_time:.4f} seconds")

        return torch.tensor(top_k_loss, device=self.device, dtype=torch.float32)

    def compute_top_k_loss_pandas(self, X, X_hat, K=10, sample_size=1000):
        """
        Compute the loss based on the difference between top-K values of input and output data
        using pandas value_counts() method.
        
        Parameters
        ----------
        X : torch.Tensor
            Original input data
        X_hat : torch.Tensor
            Reconstructed output data
        K : int
            Number of top elements to consider
        sample_size : int
            Number of samples to use for estimating top-K values
            
        Returns
        -------
        torch.Tensor
            Top-K loss value
        """
        
        # Randomly sample indices
        indices = torch.randperm(X.size(0))[:min(sample_size, X.size(0))]
        
        # Convert to numpy arrays
        X_sample = X[indices].detach().cpu().numpy()
        X_hat_sample = X_hat[indices].detach().cpu().numpy()
        
        # Initialize loss
        top_k_loss = 0
        
        # Process each column
        for col in range(X_sample.shape[1]):
            # Get top K values for input and output
            top_k_input = pd.Series(X_sample[:, col]).value_counts().nlargest(K).index.tolist()
            top_k_output = pd.Series(X_hat_sample[:, col]).value_counts().nlargest(K).index.tolist()
            
            # Calculate symmetric difference between sets
            top_k_loss += len(set(top_k_input) ^ set(top_k_output))
    
        return torch.tensor(top_k_loss, device=self.device, dtype=torch.float32)

    def loss(self, X, X_hat, mu_z, logsigma_z, beta):
        """
        Computes ELBO with additional graph structure losses for latent spaces
        """
        # Original VAE losses
        p = Normal(torch.zeros_like(mu_z), torch.ones_like(mu_z))
        q = Normal(mu_z, torch.exp(logsigma_z))
        divergence_loss = torch.sum(torch.distributions.kl_divergence(q, p))

        # Reconstruction losses
        categoric_loglik = 0
        if sum(self.num_categories) != 0:
            i = 0
            for v in range(len(self.num_categories)):
                categoric_loglik += -torch.nn.functional.cross_entropy(
                    X_hat[:, i : (i + self.num_categories[v])],
                    torch.max(X[:, i : (i + self.num_categories[v])], 1)[1],
                ).sum()
                i = i + self.decoder.num_categories[v]

        gauss_loglik = (
            Normal(
                loc=X_hat[:, -self.num_continuous:],
                scale=torch.ones_like(X_hat[:, -self.num_continuous:]),
            )
            .log_prob(X[:, -self.num_continuous:])
            .sum()
        )

        reconstruct_loss = -(categoric_loglik + gauss_loglik)

        # Graph structure losses
        latent_graph_loss = self.compute_latent_graph_loss(mu_z)  # Latent space structure
        
        # Combined loss with both graph structure terms
        total_graph_loss = latent_graph_loss
        
        # Top-K loss
        top_k_loss = self.compute_top_k_loss_pandas(X, X_hat)

        # Final loss
        elbo = reconstruct_loss + beta * divergence_loss + self.alpha * total_graph_loss

        return (
            elbo, 
            reconstruct_loss, 
            divergence_loss, 
            categoric_loglik, 
            gauss_loglik, 
            latent_graph_loss,
            top_k_loss
        )

    def generate(self, N):
        """
        Generate N synthetic samples from the learned model
        
        Parameters
        ----------
        N : int
            Number of samples to generate
            
        Returns
        -------
        torch.Tensor
            Generated samples with appropriate categorical and continuous values
        """
        z_samples = torch.randn_like(
            torch.ones((N, self.encoder.latent_dim)), device=self.device
        )
        x_gen = self.decoder(z_samples)
        x_gen_ = torch.ones_like(x_gen, device=self.device)
        i = 0

        for v in range(len(self.num_categories)):
            x_gen_[
            :, i: (i + self.num_categories[v])
            ] = torch.distributions.one_hot_categorical.OneHotCategorical(
                logits=x_gen[:, i: (i + self.num_categories[v])]
            ).sample()
            i = i + self.num_categories[v]

        x_gen_[:, -self.num_continuous:] = x_gen[
                                           :, -self.num_continuous:
                                           ]
        return x_gen_

    def save(self, filename):
        """Save model state dict to file"""
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        """Load model state dict from file"""
        self.load_state_dict(torch.load(filename))