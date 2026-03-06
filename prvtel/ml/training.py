import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from tqdm import tqdm
import visdom
import json
from opacus import GradSampleModule
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus import PrivacyEngine, GradSampleModule
import os
import pickle

# User code.
from ..config import Config
from ..models.VAE_corr import VAE as VAE_corr, RNNVAE, sigmoid, BETA_SCALE
from ..models.VAE import VAE
from ..models.graph_VAE import GraphVAE, Decoder as GraphDecoder, Encoder as GraphEncoder
from ..data.dataloaders import ThreadedChunkDataset, ChunkDataset, SequenceChunkDataset, ThreadedSequenceDataset
from .visualization import VisdomLogger


def configure_training(config, model, dataloader, rank=0, world_size=1):
    """
    Configure the given VAE model for distributed training and 
    differential privacy if necessary.

    Args:
        config: A Config object that contains

    """
    module = model

    # Apply distributed wrapper.
    # NOTE: DPDDP doesn't natively work with uneven inputs. For now just disable distributed for diff priv...
    if world_size > 1:
        # For RNN-VAE, we might need to find unused parameters due to sequence length differences
        find_unused_params = hasattr(model, 'seq_len')  # RNN-VAE has seq_len attribute
        module = DDP(module, device_ids=[rank], find_unused_parameters=find_unused_params)
        return module

    # Configure privacy engine.
    if config.differential_privacy:
        priv = PrivacyEngine()

        (
            configured_model,
            configured_optimizer,
            _
        ) = priv.make_private_with_epsilon(
            module=module,
            optimizer=model.optimizer,
            data_loader=dataloader,
            target_epsilon=config.target_eps,
            target_delta=config.target_delta,
            epochs=config.n_epochs,
            max_grad_norm=config.C,
            poisson_sampling=False,
            loss_reduction='sum'
        )
        # Attach to underlying model.
        model.optimizer = configured_optimizer
        model.privacy_engine = priv
    else:
        configured_model = module

    return configured_model


def unwrap_model(model_wrapper):
    """Unwrap the model from any distributed or privacy wrappers."""
    if isinstance(model_wrapper, GradSampleModule):
        model = model_wrapper._module
    elif isinstance(model_wrapper, DDP):
        model = model_wrapper.module
    else:
        model = model_wrapper
    return model
    

def _base_train_loop(
    vae,
    dataloader,
    n_epochs,
    logging_freq=1,
    patience=5,
    delta=10,
    filepath=None,
    rank=0,
    world_size=1,
    is_graph_vae=False,
    vis=None
):
    """Base training loop for both standard VAE and GraphVAE
    Args:
        vae: VAE or GraphVAE model
        dataloader: DataLoader providing training data
        n_epochs: Number of epochs to train
        logging_freq: How often to log metrics
        patience: Number of epochs to wait before early stopping
        delta: Minimum change in ELBO for early stopping
        filepath: Where to save best model
        rank: Process rank for distributed training
        world_size: Total number of processes for distributed training
        is_graph_vae: Whether this is training a GraphVAE
        vis: Visdom visualizer instance
    Returns:
        Best model state dict based on reconstruction loss
    """
    # log_metrics: List of values for each metric across epochs
    log_metrics = {
        'train_loss': [],
        'reconstruct_loss': [],
        'regularization_loss': [],
        'categorical_reconstruct': [],
        'numerical_reconstruct': [],
    }
    if is_graph_vae:
        log_metrics.update({
            'latent_graph_loss': [],
            'top_k_loss': []
        })

    # Early stopping setup
    min_elbo = 0.0
    stop_counter = 0
    min_reconstruction_loss = float('inf')
    logger = logging.getLogger(__name__)

    # Handle DDP and Opacus wrappers. Use model_wrapper for forward pass (DDP/Opacus hooks), and use
    # model to access any underlying attributes.
    # NOTE: DPDDP doesn't natively work with uneven inputs. For now just disable distributed for diff priv...
    model_wrapper = vae
    model = unwrap_model(model_wrapper)

    best_model_state = model.state_dict()

    # Initialize visualization
    visdom_logger = VisdomLogger(vis if rank == 0 else None, is_graph_vae)

    for epoch in range(n_epochs):
        # metrics: Single values for current epoch
        metrics = {
            'train_loss': 0.0,
            'reconstruct_loss': 0.0,
            'regularization_loss': 0.0,
            'categorical_reconstruct': 0.0,
            'numerical_reconstruct': 0.0,
        }
        if is_graph_vae:
            metrics.update({
                'latent_graph_loss': 0.0,
                'top_k_loss': 0.0
            })
        
        beta = sigmoid(epoch / n_epochs) if model.use_beta else BETA_SCALE
        logger.info(f"Epoch {epoch}, Beta: {beta}")

        # Training loop
        for idx, (Y_subset,) in enumerate(tqdm(dataloader, position=rank, desc=f'| Epoch (Rank {rank}): {epoch} |')):
            model.optimizer.zero_grad()
            X = Y_subset.to(model.device)
            X_hat, mu_z, logsigma_z = model_wrapper(X)
            
            loss_outputs = model.loss(X, X_hat=X_hat, mu_z=mu_z, logsigma_z=logsigma_z, beta=beta)
            
            elbo = loss_outputs[0] * world_size  # Compensate for DDP gradient division
            elbo.backward()
            model.optimizer.step()

            # Update metrics - map loss outputs to specific metrics
            loss_names = ['train_loss', 'reconstruct_loss', 'regularization_loss', 
                         'categorical_reconstruct', 'numerical_reconstruct']
            if is_graph_vae:
                loss_names.extend(['latent_graph_loss', 'top_k_loss'])
                
            for i, name in enumerate(loss_names):
                metrics[name] += loss_outputs[i].item()

            # TODO: Hardcoded frequency and print statements to get around multiprocess logging.
            if idx % 200 == 0 and rank == 0 and world_size > 1:
                print(f"Batch {idx} || ELBO: {loss_outputs[0].item():.2f} || Recon: {loss_outputs[1].item():.2f} || Reg: {loss_outputs[2].item():.2f}")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"Min, Max, IsNan of Gradient ({name}): {param.grad.min()}, {param.grad.max()}, {param.grad.isnan().any()}")

        # Update logs
        for key, value in metrics.items():
            log_metrics[key].append(value)

        # Model saving logic
        if metrics['reconstruct_loss'] < min_reconstruction_loss:
            min_reconstruction_loss = metrics['reconstruct_loss']
            if rank == 0 and filepath is not None:
                best_model_state = model.state_dict()
                model.save(filepath)

        # Early stopping logic
        if epoch == 0:
            min_elbo = metrics['train_loss']
        if metrics['train_loss'] < (min_elbo - delta):
            min_elbo = metrics['train_loss']
            stop_counter = 0
        else:
            stop_counter += 1

        # Logging
        if epoch % logging_freq == 0:
            _log_metrics(logger, epoch, metrics, is_graph_vae)

        # Update visualization
        visdom_logger.update(epoch, metrics, logger)

    return best_model_state

def _log_metrics(logger, epoch, metrics, is_graph_vae):
    """Log training metrics in a formatted table
    Args:
        logger: Logger instance
        epoch: Current epoch number
        metrics: Dict of metric values
        is_graph_vae: Whether to include GraphVAE specific metrics
    """
    if epoch % 10 == 0:
        header = (
            f"{'Epoch':>5} {'ELBO':>15} {'Recon':>15} {'Regularization':>15} "
            f"{'Cat Loss':>15} {'Num Loss':>15}"
        )
        if is_graph_vae:
            header += " {'Latent Graph':>15} {'Top-K':>15}"
        logger.info(header + f"\n{'-' * (75 if not is_graph_vae else 120)}")

    log_str = (
        f"{epoch:>5d} {metrics['train_loss']:>15,.2f} "
        f"{metrics['reconstruct_loss']:>15,.2f} "
        f"{metrics['regularization_loss']:>15,.2f} "
        f"{metrics['categorical_reconstruct']:>15,.2f} "
        f"{metrics['numerical_reconstruct']:>15,.2f}"
    )
    if is_graph_vae:
        log_str += (f" {metrics['latent_graph_loss']:>15,.2f} "
                   f"{metrics['top_k_loss']:>15,.2f}")
    logger.info(log_str)

def vae_train_loop(*args, **kwargs):
    return _base_train_loop(*args, **kwargs, is_graph_vae=False)

def graphvae_train_loop(*args, **kwargs):
    return _base_train_loop(*args, **kwargs, is_graph_vae=True)

def _init_base_vae(config, input_dim, num_conts, num_cats_per_col, device, 
                   encoder_class, decoder_class, vae_class, graph_prior=None):
    """Base initialization function for VAE models
    Args:
        config: Configuration object containing model parameters
        input_dim: Input dimension size
        num_conts: Number of continuous variables
        num_cats_per_col: Number of categories per categorical column
        device: Device to place model on ('gpu' or 'cpu')
        encoder_class: Encoder class to instantiate
        decoder_class: Decoder class to instantiate
        vae_class: VAE class to instantiate
        graph_prior: Optional adjacency matrix for GraphVAE
    """
    encoder = encoder_class(input_dim, config.latent_dim, hidden_dim=config.hidden_dim, device=device)
    decoder = decoder_class(config.latent_dim, num_conts, hidden_dim=config.hidden_dim, 
                          num_categories=num_cats_per_col, device=device)
    vae_kwargs = {
        'encoder': encoder,
        'decoder': decoder,
        'lr': config.learning_rate,
        'use_beta': config.use_beta,
        'regularizer_type': config.regularizer_type
    }
    if graph_prior is not None:
        vae_kwargs['graph_prior'] = graph_prior
        
    return vae_class(**vae_kwargs)

def init_model(config: Config | dict, input_dim, num_conts, num_cats_per_col, device='cuda', **kwargs):
    """Base initialization function for VAE (and potentially other) models
    Args:
        config: Either a Config object with a path to a a config file JSON or a dictionary with model parameters.
        input_dim: Input dimension size
        num_conts: Number of continuous variables
        num_cats_per_col: Number of categories per categorical column
        device: Device to place model on ('cpu' or 'cuda:{i}')
        **kwargs: Additional keyword arguments to pass to the model initializer/constructor not specified in the config.
    Returns:
        model: Initialized PyTorch model moved to the specified device.
    """

    MODELS = {
        'vae': VAE,
        'vae_corr': VAE_corr,
        'rnn_vae': RNNVAE
    }

    # Load config if necessary.
    model_config = config
    if isinstance(config, Config):
        with open(config.config_file_path) as file:
            config = json.load(file)
        model_config = config['model']

    try:
        model_initializer = MODELS[model_config['type']]
    except KeyError:
        raise ValueError(f'Unrecognized model type. Must be one of: {list(MODELS.keys())}')
    
    model_kwargs = model_config.get('model_kwargs', {})
    model = model_initializer(input_dim, num_conts, num_cats_per_col, device=device, **model_kwargs, **kwargs)
    model = model.to(device)

    return model

def save_model_init_params(
    model_configs: dict, 
    input_dim: int, 
    num_conts: int, 
    num_cats_per_col: list[int], 
    save_path: str
):
    """
    Save model initialization parameters to disk, so we can initialize a fresh model and load its state dict during inference. 
    The model initialization parameters take the form of a dictionary that can be unpacked and passed to the init_model function.
    The model_configs dictionary is the "model" key in the config file.
    """
    with open(save_path, mode='wb') as file:
        model_init_params = dict(
            config=model_configs,
            input_dim=input_dim,
            num_conts=num_conts,
            num_cats_per_col=num_cats_per_col
        )
        pickle.dump(model_init_params, file)

def init_graph_vae(config, input_dim, num_conts, num_cats_per_col, adjacency_matrix, device='gpu'):
    return _init_base_vae(
        config, input_dim, num_conts, num_cats_per_col, device,
        encoder_class=GraphEncoder,
        decoder_class=GraphDecoder,
        vae_class=GraphVAE,
        graph_prior=adjacency_matrix
    )

def init_dataloader(config, data, rank=0, world_size=1, use_threaded_dataset=True, device='cuda', 
                   dataset_type='standard', seq_len=50, stride=25):
    
    if dataset_type == 'sequence':
        # Use sequence datasets for RNN-VAE
        if use_threaded_dataset:
            dataset = ThreadedSequenceDataset(
                ddf=data,
                batch_size=config.batch_size,
                seq_len=seq_len,
                stride=stride,
                device=device,
                cache_size=config.num_chunks_cached,
                qsize=1
            )
        else:
            dataset = SequenceChunkDataset(
                ddf=data,
                batch_size=config.batch_size,
                seq_len=seq_len,
                stride=stride,
                cache_size=config.num_chunks_cached,
                device=device
            )
    else:
        # Use standard datasets for regular VAE
        if use_threaded_dataset:
            dataset = ThreadedChunkDataset(
                ddf=data,
                batch_size=config.batch_size,
                device=device,
                cache_size=config.num_chunks_cached,
                qsize=1
            )
        else:
            dataset = ChunkDataset(
                ddf=data,
                batch_size=config.batch_size,
                cache_size=config.num_chunks_cached,
                device=device
            )
    # Shard for multiprocess.
    if world_size > 1:
        dataset = dataset.create_worker_split(rank, world_size)
    # Should we use pinned memory?
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
        pin_memory=False
    )
    return dataloader