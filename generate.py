# Standard imports
import torch
from prvtel.config import Config
import os
import argparse
import logging
from datetime import datetime
import sys
import glob
import json
from dask.distributed import Client
import pickle

# User defined utility functions.
from prvtel.ml.training import init_model
from prvtel.ml.inference import generate_synthetic_traces

import warnings
warnings.filterwarnings("ignore")


def get_command_line_args():
    # Initialize
    parser = argparse.ArgumentParser(description='Evaluate synthetic data quality with Jensen-Shannon distance (categorical features) '
                                     'or Earth Mover\'s distance (continuous features).')

    # Paths.
    parser.add_argument('--model_path', required=True, help='Path to model')
    parser.add_argument('--preprocessor_path', required=True, help='Path to the preprocessors used when training the model.')
    parser.add_argument('--syn_data_path', type=str, required=True,
                          help='Path to save synthetic data')
    
    # Tuning the generation.
    parser.add_argument('--generation_size', type=int, default=None,
                          help='Number of synthetic data points to generate in total.')
    parser.add_argument('--batch_size', type=int, default=None,
                          help='Number of samples to generate at a time.')
    parser.add_argument('--num_parts', default=None, help='Number of chunks to write to disk')
    parser.add_argument('--single_file', action='store_true', default=False,
                          help='Whether the generated data should go to a single file.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_command_line_args()

    client = Client()
    print(f'Dashboard Link: {client.dashboard_link}')
    
    # Create synthetic data save directory if it doesn't exist
    print('Creating directories to save generated data...')
    os.makedirs(os.path.dirname(args.syn_data_path), exist_ok=True)

    # Load metadata.
    with open(args.preprocessor_path, mode='rb') as file:
        preprocessing_info = pickle.load(file)
    # Load model init params.
    directory = os.path.dirname(args.model_path)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    model_init_path = os.path.join(directory, f'{model_name}_init_params.pkl')
    with open(model_init_path, mode='rb') as file:
        model_init_params = pickle.load(file)

    # Preprocessors
    transforms = preprocessing_info['transforms']

    # Data needed to generate data and reverse transformations.
    metadata = preprocessing_info['reference_data']
    original_cols = metadata['original_cols']
    reordered_cols = metadata['reordered_cols']
    # If these are not provided, use those that were used during the training of the model.
    num_parts = metadata['npartitions']
    batch_size = args.batch_size if args.batch_size else metadata['batch_size']
    size = args.generation_size if args.generation_size else metadata['nrows']

    if size is None:
        raise ValueError('No generation size provided and unable to infer from metadata. Please provide a generation size.')

    generate_synthetic_traces(
        model_path=args.model_path,
        model_init_params=model_init_params,
        syn_data_path=args.syn_data_path,
        transforms=transforms,
        batch_size=batch_size,
        size=size,
        reordered_cols=reordered_cols,
        output_cols=original_cols,
        num_parts=num_parts,
        single_file=args.single_file
    )