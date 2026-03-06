from random import gauss
from pandas import Categorical
import torch
import torch.nn as nn
import numpy as np
from opacus import PrivacyEngine
from .regularizers import KLDivergenceRegularizer, MMDRegularizer

# from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal

from tqdm import tqdm
import logging

BETA_SCALE = 0.01

def sigmoid(x):
    # Custom sigmoid function that scales the output to range [0, 0.01]
    # Used for beta annealing in VAE training
    return 1 / (1 + np.exp(-x)) * BETA_SCALE

class Encoder(nn.Module):
    """Encoder, takes in x
    and outputs mu_z, sigma_z
    (diagonal Gaussian variational posterior assumed)
    """

    def __init__(
        self, input_dim, latent_dim, hidden_dim=32, activation=nn.Tanh
    ):
        super().__init__()
        output_dim = 2 * latent_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
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


class RNNEncoder(nn.Module):
    """RNN-based encoder for temporal sequences
    Takes in sequences of shape (batch_size, seq_len, input_dim)
    Outputs mu_z, sigma_z of shape (batch_size, latent_dim)
    """

    def __init__(self, input_dim, latent_dim, hidden_dim=32, rnn_type='LSTM'):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Simple RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Simple output layer
        self.output_net = nn.Linear(hidden_dim, 2 * latent_dim)

    def forward(self, x):
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # If input is 2D (batch_size, input_dim), treat as single timestep
            # Reshape to (batch_size, 1, input_dim)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected input to be 2D or 3D, got {x.dim()}D tensor")
        
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Use the last time step's output
        # rnn_out shape: (batch_size, seq_len, hidden_dim)
        last_output = rnn_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
        
        # Project to latent parameters
        outs = self.output_net(last_output)
        mu_z = outs[:, :self.latent_dim]
        logsigma_z = outs[:, self.latent_dim:]
        
        return mu_z, logsigma_z


class Decoder(nn.Module):
    """Decoder, takes in z and outputs reconstruction"""

    def __init__(
        self,
        latent_dim,
        num_continuous,
        num_categories=[0],
        hidden_dim=32,
        activation=nn.Tanh,
    ):
        super().__init__()

        self.num_continuous = num_continuous
        self.num_categories = num_categories

        # Base network for shared features
        self.base_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            
        )
        
        # For categorical outputs
        categorical_output_dim = sum(num_categories)
        if categorical_output_dim > 0:
            self.categorical_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, categorical_output_dim),
            )
        
        # For continuous outputs - mean vector
        self.continuous_mean_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, num_continuous),
        )
        
        # For continuous outputs - covariance matrix (lower triangular elements)
        # We need n*(n+1)/2 parameters for an nxn covariance matrix
        if num_continuous > 0:
            cov_params = (num_continuous * (num_continuous + 1)) // 2
            self.continuous_cov_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, cov_params),
            )


    def forward(self, z):
        base_features = self.base_net(z)
        outputs = []
        
        # Process categorical features
        if sum(self.num_categories) > 0:
            categorical_outputs = self.categorical_net(base_features)
            outputs.append(categorical_outputs)
        
        # Process continuous features - mean and covariance parameters
        if self.num_continuous > 0:
            means = self.continuous_mean_net(base_features)
            
            # Get lower triangular elements of covariance matrix
            cov_params = self.continuous_cov_net(base_features)
            
            # We'll store these together for easier handling in the loss function
            outputs.append(means)
            outputs.append(cov_params)
            
        # Return all outputs (categorical logits, continuous means, and covariance params)
        return outputs


class SimpleRNNDecoder(nn.Module):
    """Simple RNN decoder for numerical data only"""

    def __init__(self, latent_dim, output_dim, seq_len, hidden_dim=32, rnn_type='LSTM'):
        super().__init__()
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        
        # Output layers with better distribution modeling
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.logvar_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        batch_size = z.size(0)
        
        # Repeat latent vector for each timestep
        z_seq = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch_size, seq_len, latent_dim)
        
        # RNN forward
        rnn_out, _ = self.rnn(z_seq)  # (batch_size, seq_len, hidden_dim)
        
        # Generate mean and variance for each timestep
        mean = self.mean_layer(rnn_out)  # (batch_size, seq_len, output_dim)
        logvar = self.logvar_layer(rnn_out)  # (batch_size, seq_len, output_dim)
        
        # Sample from distribution during training for better variance
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            output = mean + eps * std
        else:
            output = mean
            
        return output, mean, logvar

class VAE(nn.Module):
    """Combines encoder and decoder into full VAE model
    
    Parameters
    ----------
    encoder : Encoder
        Neural network that encodes input data into latent space representations
    decoder : Decoder
        Neural network that decodes latent representations back to input space
    lr : float, default=5e-4
        Learning rate for the Adam optimizer
    use_beta : bool, default=True
        Whether to use beta-VAE formulation with annealing schedule for KL term
        If True, beta increases from ~0 to 0.01 over training
        If False, beta is fixed at 0.01
    
    Attributes
    ----------
    device : torch.device
        Device (CPU/GPU) where the model runs, inherited from encoder
    num_categories : list
        List containing number of categories for each categorical variable
    num_continuous : int
        Number of continuous variables in the data
    optimizer : torch.optim.Adam
        Adam optimizer for model parameters
    """
    def __init__(
            self,
            input_dim,  
            num_conts, 
            num_cats_per_col, 
            latent_dim=8, 
            hidden_dim=32, 
            regularizer_type="kl", 
            lr=5e-4, 
            use_beta=False,
            device='cuda'
        ):
        super().__init__()

        encoder = Encoder(input_dim, latent_dim, hidden_dim)
        decoder = Decoder(latent_dim, num_conts, num_categories=num_cats_per_col, 
                          hidden_dim=hidden_dim)

        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.encoder = encoder
        self.decoder = decoder
        self.num_categories = self.decoder.num_categories
        self.num_continuous = self.decoder.num_continuous
        self.use_beta = use_beta
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr = lr

        self.to(self.device)
        print(f"Covariance VAE running on {self.device}")
        
        logger = logging.getLogger(__name__)
        # Initialize regularizer based on type
        if regularizer_type.lower() == "kl":
            logger.info("Using KL divergence regularizer")
            self.regularizer = KLDivergenceRegularizer(self.device)
        elif regularizer_type.lower() == "mmd":
            logger.info("Using MMD regularizer")
            self.regularizer = MMDRegularizer(self.device)
        else:
            raise ValueError(f"Unknown regularizer type: {regularizer_type}")

    def forward(self, X):
        """
        Encodes and then decodes a batch of data points.
        Returns a single tensor for compatibility with distributed training.
        """
        mu_z, logsigma_z = self.encoder(X)

        #######################
        # SAMPLING LAYER
        # Reparameterization trick to allow backpropagation through sampling:
        # Instead of sampling directly from N(mu_z, sigma_z), we sample from N(0,1) and transform
        s = torch.randn_like(mu_z)  # Sample from standard normal N(0,1)
        z_samples = mu_z + s * torch.exp(logsigma_z)  # Transform to sample from N(mu_z, sigma_z)
        #######################

        # Decode the sampled latent vectors
        decoder_outputs = self.decoder(z_samples)
        
        # For backward compatibility with the training loop, combine the outputs into a single tensor
        # This ensures all model parameters are used in loss calculation
        combined_tensor = self._combine_decoder_outputs(decoder_outputs, X.shape[1])
        
        return combined_tensor, mu_z, logsigma_z

    def _combine_decoder_outputs(self, decoder_outputs, output_dim):
        """
        Combines the list of decoder outputs into a single tensor for compatibility with distributed training.
        """
        combined = torch.zeros((decoder_outputs[0].shape[0], output_dim), device=self.device)
        
        # Process categorical outputs
        if sum(self.num_categories) > 0:
            categorical_logits = decoder_outputs[0]
            combined[:, :sum(self.num_categories)] = categorical_logits
        
        # Process continuous outputs
        if self.num_continuous > 0:
            means = decoder_outputs[-2]
            combined[:, -self.num_continuous:] = means
        
        return combined

    def generate(self, N):
        """
        Generate N samples from the VAE.
        Returns a tensor of shape (N, feature_dim)
        """
        # Sample from the latent space
        z_samples = torch.randn((N, self.encoder.latent_dim), device=self.device)
        
        # Get decoder outputs
        decoder_outputs = self.decoder(z_samples)
        
        # Initialize result tensor
        generated_sample = []
        
        # Handle categorical variables
        if sum(self.num_categories) > 0:
            categorical_logits = decoder_outputs[0]
            i = 0
            cat_outputs = []
            for v in range(len(self.num_categories)):
                # Sample from categorical distribution
                cat_dist = torch.distributions.one_hot_categorical.OneHotCategorical(
                    logits=categorical_logits[:, i:(i + self.num_categories[v])]
                )
                cat_outputs.append(cat_dist.sample())
                i += self.num_categories[v]
            
            if cat_outputs:
                generated_sample.append(torch.cat(cat_outputs, dim=1))
        
        # Handle continuous variables - faster batched implementation
        if self.num_continuous > 0:
            means = decoder_outputs[-2]
            
            # For generation, we can use a simplified approach that preserves correlations
            # First we'll generate uncorrelated noise
            eps = torch.randn_like(means)
            
            # Use batch matrix operations to add correlated noise efficiently
            # First reshape continuous means for batch matrix operations
            N = means.shape[0]
            
            # Option 1: Fast approximation - use sample covariance from mean predictions
            # This will produce correlated samples without the expensive per-sample operations
            centered_means = means - means.mean(dim=0, keepdim=True)
            cov = torch.matmul(centered_means.T, centered_means) / (N - 1)
            
            # Add small diagonal term for stability
            cov = cov + torch.eye(self.num_continuous, device=self.device) * 1e-4
            
            # Compute Cholesky decomposition once for the batch
            try:
                L = torch.linalg.cholesky(cov)
                
                # # Apply correlation to the noise
                # correlated_noise = torch.matmul(eps, L.T)
                
                # Scale the noise (less noise for more stable generation)
                noise_scale = 0.01  # Reduced from 1.0 to make generation more stable
                continuous_samples = means
                
            except Exception as e:
                # Fallback to uncorrelated noise if Cholesky fails
                print(f"Warning: Using uncorrelated noise for generation due to: {e}")
                continuous_samples = means
            
            generated_sample.append(continuous_samples)
        
        # Combine all generated features
        return torch.cat(generated_sample, dim=1)
    
    def loss(self, X, X_hat, mu_z, logsigma_z, beta):
        """ 
        Computes the total loss with correlated features
        
        Parameters:
        - X: Original input data
        - X_hat: Combined tensor from forward method
        - mu_z: Mean of the latent space
        - logsigma_z: Log standard deviation of the latent space
        - beta: Weight for the regularization term
        """
        # Get decoder outputs to match X_hat structure 
        # This ensures we can still work with our decoder outputs internally
        batch_size = X.shape[0]

        decoder_outputs = self.decoder(mu_z + torch.randn_like(mu_z) * torch.exp(logsigma_z))
        # print(mu_z, logsigma_z)
        # print(mu_z.isnan().any(), logsigma_z.isnan().any())
        # print(mu_z.isinf().any(), logsigma_z.isinf().any())
        # print(mu_z.max(), logsigma_z.max())
        # print(mu_z.min(), logsigma_z.min())
        
        # Compute regularization loss using the chosen regularizer
        regularization_loss = self.regularizer.compute_loss(mu_z, logsigma_z) / batch_size

        reconstruct_loss = 0
        categoric_loglik = torch.Tensor([0]).to(self.device)
        gauss_loglik = torch.Tensor([0]).to(self.device)
        
        # Process categorical variables
        if sum(self.num_categories) > 0:
            categorical_logits = decoder_outputs[0]
            i = 0
            for v in range(len(self.num_categories)):
                categoric_loglik += -torch.nn.functional.cross_entropy(
                    categorical_logits[:, i:(i + self.num_categories[v])],
                    torch.max(X[:, i:(i + self.num_categories[v])], 1)[1],
                )
                i += self.num_categories[v]
            
            # Add correlation penalty for categorical variables
            # For categorical data, we can measure associations between variables
            # using contingency tables and normalized mutual information
            cat_data = X[:, :sum(self.num_categories)]
            
            # Get one-hot predictions (using argmax)
            cat_pred = torch.zeros_like(cat_data)
            start_idx = 0
            for v, num_cat in enumerate(self.num_categories):
                if num_cat > 0:
                    end_idx = start_idx + num_cat
                    cat_logits = categorical_logits[:, start_idx:end_idx]
                    cat_indices = torch.argmax(cat_logits, dim=1)
                    # Convert to one-hot
                    cat_pred[:, start_idx:end_idx] = torch.nn.functional.one_hot(
                        cat_indices, num_classes=num_cat).float()
                    start_idx = end_idx
            
            # Calculate association matrices for real and predicted categorical data
            # We'll use the dot product of centered one-hot encodings as a simple measure
            if len(self.num_categories) > 1:  # Only needed if we have multiple categorical variables
                # Center the categorical data (subtract mean of each one-hot encoded feature)
                centered_cat_real = cat_data - cat_data.mean(dim=0)
                centered_cat_pred = cat_pred - cat_pred.mean(dim=0)
                
                # Compute association matrices
                assoc_real = torch.matmul(centered_cat_real.T, centered_cat_real) / (cat_data.shape[0] - 1)
                assoc_pred = torch.matmul(centered_cat_pred.T, centered_cat_pred) / (cat_pred.shape[0] - 1)
                
                # Association matrix difference penalty
                cat_corr_penalty = torch.nn.functional.mse_loss(assoc_real, assoc_pred) * sum(self.num_categories)
                
                # Add to categorical log-likelihood with a modest weight
                categoric_loglik -= 10 * cat_corr_penalty  # Use minus here since we want to reduce the log-likelihood
        
        # Process continuous variables
        if self.num_continuous > 0:
            cont_data = X[:, -self.num_continuous:]
            means = decoder_outputs[-2]
            cov_params = decoder_outputs[-1]
            
            # Basic MSE loss
            cont_mse = torch.nn.functional.mse_loss(means, cont_data)
            
            # Add correlation penalty
            centered_real = cont_data - cont_data.mean(dim=0)
            centered_pred = means - means.mean(dim=0)
            
            # Compute correlation matrices
            corr_real = torch.matmul(centered_real.T, centered_real) / (cont_data.shape[0] - 1)
            corr_pred = torch.matmul(centered_pred.T, centered_pred) / (means.shape[0] - 1)
            
            # Correlation matrix difference penalty
            corr_penalty = torch.nn.functional.mse_loss(corr_real, corr_pred) * self.num_continuous
            
            # Ensure all parameters are used in the loss by incorporating covariance parameters
            cov_regularization = torch.mean(cov_params**2) * 0.0001  # Very small weight
            
            # Combine the reconstruction loss with the correlation penalty
            # Give more weight to the correlation penalty
            gauss_loglik = -cont_mse - 10 * corr_penalty - cov_regularization
        
        reconstruct_loss = -(categoric_loglik + gauss_loglik)
                        
        # Total loss (ELBO)
        total_loss = reconstruct_loss + beta * regularization_loss

        return (total_loss, reconstruct_loss, regularization_loss, categoric_loglik, gauss_loglik)

    def get_privacy_spent(self, delta):
        if hasattr(self, "privacy_engine"):
            return self.privacy_engine.get_privacy_spent(delta)
        else:
            print(
                """This VAE object does not a privacy_engine attribute.
                Run diff_priv_train to create one."""
            )

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


class RNNVAE(nn.Module):
    """Simple RNN-VAE for numerical time series data"""
    
    def __init__(
            self,
            input_dim,  
            num_conts, 
            num_cats_per_col, 
            seq_len=50,
            latent_dim=8, 
            hidden_dim=32,
            rnn_type='LSTM',
            regularizer_type="kl", 
            lr=5e-4, 
            use_beta=False,
            device='cuda',
            **kwargs  # Ignore extra parameters
        ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.use_beta = use_beta
        
        # Simple components
        self.encoder = RNNEncoder(input_dim, latent_dim, hidden_dim, rnn_type=rnn_type)
        self.decoder = SimpleRNNDecoder(latent_dim, input_dim, seq_len, hidden_dim, rnn_type)
        
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr = lr

        self.to(self.device)
        print(f"Simple RNN-VAE running on {self.device}")
        
        logger = logging.getLogger(__name__)
        if regularizer_type.lower() == "kl":
            self.regularizer = KLDivergenceRegularizer(self.device)
        else:
            raise ValueError(f"Unknown regularizer type: {regularizer_type}")
            
        # For compatibility
        self.num_continuous = num_conts
        self.num_categories = [0]

    def forward(self, X):
        """Simple forward pass"""
        # Handle 2D input (convert to 3D)
        if X.dim() == 2:
            X = X.unsqueeze(1)  # (batch, 1, features)
        
        mu_z, logsigma_z = self.encoder(X)
        
        # Reparameterization trick
        s = torch.randn_like(mu_z)
        z_samples = mu_z + s * torch.exp(logsigma_z)
        
        # Decode - now returns output, mean, logvar
        X_hat, mean, logvar = self.decoder(z_samples)
        
        # If input was 2D, return 2D
        if X.shape[1] == 1:
            X_hat = X_hat.squeeze(1)
        
        return X_hat, mu_z, logsigma_z

    def _combine_sequence_outputs(self, decoder_outputs, feature_dim):
        """
        Combines sequence outputs into a single tensor for compatibility
        decoder_outputs: list of length seq_len, each containing outputs for that timestep
        """
        batch_size = decoder_outputs[0][0].shape[0] if len(decoder_outputs) > 0 else 0
        
        # Create combined tensor of shape (batch_size, seq_len, feature_dim)
        combined = torch.zeros((batch_size, self.seq_len, feature_dim), device=self.device)
        
        for t in range(self.seq_len):
            timestep_outputs = decoder_outputs[t]
            
            # Process categorical outputs
            if sum(self.num_categories) > 0:
                categorical_logits = timestep_outputs[0]
                combined[:, t, :sum(self.num_categories)] = categorical_logits
            
            # Process continuous outputs
            if self.num_continuous > 0:
                means = timestep_outputs[-2]
                combined[:, t, -self.num_continuous:] = means
        
        # If we need to return 2D for compatibility with non-sequence training
        # Take the last timestep or flatten the sequence dimension
        if combined.shape[1] == 1:
            # Single timestep case - return 2D
            combined = combined.squeeze(1)  # (batch_size, feature_dim)
        
        return combined

    def generate(self, N):
        """Generate N samples from the simplified RNN-VAE"""
        # Sample from latent space
        z_samples = torch.randn((N, self.encoder.latent_dim), device=self.device)
        
        # Set to eval mode to get mean predictions without noise
        self.eval()
        with torch.no_grad():
            # Decode to get generated data
            generated_data, mean, logvar = self.decoder(z_samples)  # (N, seq_len, input_dim)
        self.train()
        
        # For single timestep generation, return 2D tensor
        if self.seq_len == 1 or generated_data.shape[1] == 1:
            return generated_data.squeeze(1)  # (N, input_dim)
        
        # For multi-timestep, return the last timestep (or you could return full sequence)
        return generated_data[:, -1, :]  # (N, input_dim)
    
    def loss(self, X, X_hat, mu_z, logsigma_z, beta):
        """Proper probabilistic loss"""
        batch_size = X.shape[0]

        # Regularization loss
        regularization_loss = self.regularizer.compute_loss(mu_z, logsigma_z) / batch_size

        # Get decoder mean and logvar
        z_samples = mu_z + torch.randn_like(mu_z) * torch.exp(logsigma_z)
        _, mean, logvar = self.decoder(z_samples)

        # Proper Gaussian negative log-likelihood
        # Instead of MSE + variance penalties, use proper likelihood
        variance = torch.exp(logvar) + 1e-8
        nll = 0.5 * (torch.log(2 * np.pi * variance) + (X - mean)**2 / variance)
        reconstruct_loss = nll.sum()

        total_loss = reconstruct_loss + beta * regularization_loss

        # For compatibility
        categoric_loglik = torch.Tensor([0]).to(self.device)
        gauss_loglik = -reconstruct_loss

        return (total_loss, reconstruct_loss, regularization_loss, categoric_loglik, gauss_loglik)

    def get_privacy_spent(self, delta):
        if hasattr(self, "privacy_engine"):
            return self.privacy_engine.get_privacy_spent(delta)
        else:
            print(
                """This RNN-VAE object does not have a privacy_engine attribute.
                Run diff_priv_train to create one."""
            )

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
