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
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        outs = self.net(x)
        mu_z = outs[:, : self.latent_dim]
        logsigma_z = outs[:, self.latent_dim :]
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

        output_dim = num_continuous + sum(num_categories)
        self.num_continuous = num_continuous
        self.num_categories = num_categories

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)

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
        print(f"VAE running on {self.device}")
        
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
        X_hat = self.decoder(z_samples)
        return X_hat, mu_z, logsigma_z


    def generate(self, N):
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
    
    def loss(self, X, X_hat, mu_z, logsigma_z, beta):
        """ 
        Computes the total loss consisting of reconstruction loss and regularization loss.
        
        Parameters
        ----------
        X : torch.Tensor
            The input data batch
        X_hat : torch.Tensor
            The reconstructed data from the decoder
        mu_z : torch.Tensor
            Mean vectors in latent space
        logsigma_z : torch.Tensor
            Log standard deviation vectors in latent space
        beta : float, optional
            Unused parameter kept for backward compatibility
            
        Returns
        -------
        tuple
            - total_loss : Combined reconstruction and regularization loss
            - reconstruct_loss : Loss measuring reconstruction quality
            - regularization_loss : KL divergence or MMD regularization term
            - categoric_loglik : Cross entropy loss for categorical features
            - gauss_loglik : Gaussian log likelihood for continuous features
        
        Notes
        -----
        The reconstruction loss combines:
        - Cross entropy loss for categorical variables
        - Gaussian log likelihood for continuous variables
        
        The regularization loss is computed using either KL divergence or MMD
        based on the regularizer specified during initialization.
        """
        # Compute regularization loss using the chosen regularizer
        regularization_loss = self.regularizer.compute_loss(mu_z, logsigma_z)

        # Have to do this, so we can call item when there are no categoricals.
        categoric_loglik = torch.Tensor([0]).to(self.device)
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