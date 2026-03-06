import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.join import Join
from dask.distributed import Client, get_client
from dask_pytorch_ddp import dispatch
from typing import Callable
import logging
from contextlib import nullcontext
import visdom
from datetime import datetime

# User code.
from .training import vae_train_loop, init_dataloader, configure_training, init_model
from ..config import Config


def train_model(config, X_train, num_conts, num_cats_per_col):
    logger = logging.getLogger(__name__)

    # Distributed setup.
    try:
        client = get_client()
    except:
        client = Client()

    # Need to make sure there are enough available processes/threads.
    num_processes = len(client.ncores())
    num_threads = sum(client.ncores().values())
    num_gpus = torch.cuda.device_count()

    logger.info(f'Total Processes: {num_processes}')
    logger.info(f'Total Threads: {num_threads}')
    logger.info(f'Total CUDA Devices: {num_gpus}')

    # Each GPU gets one and only one process.
    world_size = min(num_processes, num_gpus)
    # NOTE: DPDDP doesn't natively work with uneven inputs. For now just disable distributed for diff priv...
    if config.differential_privacy:
        world_size = 1
    # Each threaded dataset needs two cores, otherwise training will hang.
    use_threaded_dataset = (2 * world_size <= num_threads)

    logger.info(f'World Size: {world_size}')
    logger.info(f'Use Threads: {use_threaded_dataset}')

    futures = run(
        client,
        train_on_device,
        target_workers=list(range(world_size)),
        config=config,
        data=X_train,
        num_conts=num_conts,
        num_cats_per_col=num_cats_per_col,
        use_threaded_dataset=use_threaded_dataset
    )
    # Get model state from rank 0 process.
    results = client.gather(futures)
    state_dict = results[0]

    logger.info(f'Best model state dictionary saved at: {config.model_save_path}')

    # Load model state.
    model = init_model(config, X_train.shape[1], num_conts, num_cats_per_col)
    model.load_state_dict(state_dict)
    return model

def train_on_device(
    config: Config,  
    data,
    num_conts,
    num_cats_per_col,
    use_threaded_dataset=True,
):
    """
    Function run that handles training on each process. Shards underlying data and 
    pins worker to corresponding device.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank}'

    model = init_model(config=config, input_dim=data.shape[1], num_conts=num_conts, 
                       num_cats_per_col=num_cats_per_col, device=device)

    # TODO: Implement logging in multiprocess setting.
    print(model)

    # Read data configuration from config file if it exists
    if hasattr(config, 'config_file_path'):
        import json
        with open(config.config_file_path) as file:
            full_config = json.load(file)
        data_config = full_config.get('data', {})
        dataset_type = data_config.get('dataset_type', 'standard')
        seq_len = data_config.get('seq_len', 50)
        stride = data_config.get('stride', 25)
    else:
        # Fallback to config attributes
        dataset_type = getattr(config, 'dataset_type', 'standard')
        seq_len = getattr(config, 'seq_len', 50)
        stride = getattr(config, 'stride', 25)
    
    
    # Data loading.
    dataloader = init_dataloader(
        config=config,
        data=data,
        rank=rank,
        world_size=world_size,
        use_threaded_dataset=use_threaded_dataset,
        device=device,
        dataset_type=dataset_type,
        seq_len=seq_len,
        stride=stride
    )

    # Configure for data parallelism or differential privacy.
    ddp = configure_training(
        config=config,
        model=model,
        dataloader=dataloader,
        rank=rank,
        world_size=world_size
    )

    # Use Join context manager for DDP (prevents hanging w/ uneven inputs).
    context = Join([ddp]) if isinstance(ddp, DDP) else nullcontext()
    with context:
        model_state_dict = vae_train_loop(
            vae=ddp,
            dataloader=dataloader,
            n_epochs=config.n_epochs,
            logging_freq=config.logging_freq,
            patience=config.patience,
            delta=config.delta,
            filepath=config.model_save_path,
            rank=rank,
            world_size=world_size
        )
    return model_state_dict


def run(
    client: Client,
    pytorch_function: Callable,
    target_workers: list[int] | None = None,
    *args,
    backend: str = "nccl",
    pass_local_rank: bool = False,
    **kwargs
):
    """
    Dispatch a pytorch function over a dask cluster, and returns a list of futures
    for the resulting tasks.

    Modified from dask-pytorch-ddp to be able to specify which processes to schedule the
    training on by rank. Useful for scheduling training on processes whose ranks correspond
    to a CUDA devices.
    """
    # Information for each worker as list of dicts in increasing order of rank.
    all_workers = dispatch._get_worker_info(client)
    world_size = len(all_workers)
    port = 23456  # pick a free port?
    host = all_workers[0]["host"]

    # Only schedule on select processes.
    if target_workers is not None:
        target_ranks = list(set(target_workers))
        all_workers = [all_workers[rank] for rank in target_ranks]
        world_size = len(all_workers)

    # Submit training jobs and collect return values (results).
    futures = []
    for worker in all_workers:
        if pass_local_rank:
            fut = client.submit(
                dispatch.dispatch_with_ddp,
                pytorch_function=pytorch_function,
                master_addr=host,
                master_port=port,
                rank=worker["global_rank"],
                world_size=world_size,
                *args,
                local_rank=worker["local_rank"],
                backend=backend,
                workers=[worker["worker"]],
                **kwargs
            )
        else:
            fut = client.submit(
                dispatch.dispatch_with_ddp,
                pytorch_function=pytorch_function,
                master_addr=host,
                master_port=port,
                rank=worker["global_rank"],
                world_size=world_size,
                *args,
                backend=backend,
                workers=[worker["worker"]],
                **kwargs
            )
        futures.append(fut)
    return futures
