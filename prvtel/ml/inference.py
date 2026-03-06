import torch
from torch.utils.data import DataLoader
import dask.dataframe as dd
from dask.distributed import get_client, Client, fire_and_forget, wait
import pandas as pd
from tqdm import tqdm
import os
import math

# User code.
from .training import init_model
from .distributed import run


def generate_synthetic_traces(
    model_path, 
    model_init_params,
    syn_data_path,
    batch_size,
    size,
    transforms,
    reordered_cols,
    output_cols,
    num_parts=1,
    single_file=False
):
    # Distributed setup.
    try:
        client = get_client()
    except:
        client = Client()

    # Need to have one process per GPU.
    world_size = 1
    if not single_file:
        num_processes = len(client.ncores())
        num_gpus = torch.cuda.device_count()
        world_size = min(num_processes, num_gpus)

    print(f'World Size: {world_size}')

    # Find number of parts/samples to give per process.
    max_parts = math.ceil(num_parts // world_size)
    max_rows = math.ceil(size // world_size)

    futures = []
    parts_left = num_parts
    rows_left = size
    for dev in range(world_size):
        parts = max_parts if parts_left - max_parts >= 0 else parts_left
        rows = max_rows if rows_left - max_rows >= 0 else rows_left
        fut = client.submit(
            generate_large_data,
            model_path=model_path,
            model_init_params=model_init_params,
            syn_data_path=syn_data_path,
            batch_size=batch_size,
            size=rows,
            transforms=transforms,
            reordered_cols=reordered_cols,
            output_cols=output_cols,
            num_parts=parts,
            single_file=single_file,
            device=dev
        )
        futures.append(fut)
    _ = client.gather(futures)


def generate_large_data(
    model_path,
    model_init_params,
    syn_data_path,
    batch_size,
    size,
    transforms,
    reordered_cols,
    output_cols,
    num_parts=1,
    single_file=True,
    device=0
):
    # Write to files asynchronously.
    futures = []
    if not single_file:
        try:
            client = get_client()
        except:
            client = Client()

    # Configure device.
    torch.cuda.set_device(device)

    # TODO: Like with the train_model, generalize process of initializing a model.
    model = init_model(**model_init_params)
    model.load_state_dict(torch.load(model_path))
    model.to(f'cuda:{device}')

    # Generate even size chunks batch by batch.
    max_chunk_size = size // num_parts
    part_num = 0
    for i in tqdm(range(0, size, max_chunk_size), desc='Data Generation'):
        # Generate synthetic data with X_train
        chunk_size = min(max_chunk_size, size - i)
        output = _generate_chunk(
            model=model,
            transforms=transforms,
            chunk_size=chunk_size,
            max_batch_size=batch_size,
            reordered_cols=reordered_cols,
            output_cols=output_cols
        )
        if single_file:
            output_file_path = syn_data_path
            if i == 0:
                output.to_csv(output_file_path, index=False, header=True)
            else:
                output.to_csv(output_file_path, index=False, header=False, mode='a')
        else:
            output_prefix = os.path.splitext(os.path.basename(syn_data_path))[0]
            output_dir = os.path.dirname(syn_data_path)
            output_file_path = os.path.join(output_dir, f'{output_prefix}_{device}_{part_num}.csv')
            fut = client.submit(output.to_csv, output_file_path, index=False)
            futures.append(fut)
            part_num += 1
    if futures:
        wait(futures)


def invert_transforms(data, transforms=[], output_cols=None):
    # Transforms appear in order applied, so we must go backwards to invert them.
    reversed_transforms = reversed(transforms)
    output = data
    for transform in reversed_transforms:
        output = transform.inverse_transform(output)
    if output_cols is not None:
        output = output[output_cols]
    return output


def _generate_chunk(
        model,
        transforms, 
        chunk_size, 
        max_batch_size,
        reordered_cols,
        output_cols
    ):
    # Generate synthetic data batch by batch.
    batch_tensors = []
    for j in range(0, chunk_size, max_batch_size):
        batch_size = min(max_batch_size, chunk_size - j)
        with torch.inference_mode():
            batch = model.generate(batch_size)
        batch_tensors.append(batch)

    # Aggregate batches into chunk and move to CPU for writing.
    syn_chunk_tensor = torch.concat(batch_tensors)
    syn_chunk = pd.DataFrame(
        syn_chunk_tensor.cpu().numpy(),
        columns=reordered_cols
    )
    # Inverting must be done on CPU.
    output = invert_transforms(
        data=syn_chunk, 
        transforms=transforms, 
        output_cols=output_cols
    )
    return output
