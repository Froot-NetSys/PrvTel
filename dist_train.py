"""
Distributed Training Script

Example usage:
    python dist_train.py --status train --batch_size 32 --n_epochs 200 --input_data_path "data/my_dataset.csv"
    python dist_train.py --status load --input_data_path "data/my_dataset.csv"

Available arguments:
    --status            : 'train' or 'load'
    --pre_proc_method  : Preprocessing method
    --batch_size       : Batch size for training
    --latent_dim      : Latent dimension size
    --hidden_dim      : Hidden dimension size
    --n_epochs        : Number of training epochs
    --input_data_path : Path to input data file
    --differential_privacy : Enable differential privacy (True/False)
"""

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
import dask
from dask.distributed import Client
import dask.dataframe as dd
import pickle

# User defined utility functions.
from prvtel.ml.distributed import train_model
from prvtel.ml.training import save_model_init_params
from prvtel.data.preprocessing import read_large_data, preprocess_large_data

import warnings
warnings.filterwarnings("ignore")


def configure_logging(config):
    # Set up logging first, before any logger calls
    # Create results directory if it doesn't exist
    os.makedirs(config.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create log filename in results directory
    log_params = [
        f"bs_{config.batch_size}",
        f"ep_{config.n_epochs}",
        f"dp_{config.differential_privacy}",
        timestamp
    ]
    log_filename = os.path.join(config.results_dir, f"train_{'_'.join(log_params)}.log")
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Define formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    
    # Configure stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler],
        force=True
    )


def main(config: Config):
    # Get logger
    configure_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("Starting program execution...")

    # Print and log configuration
    logger.info("Configuration:")
    for key, value in vars(config).items():
        logger.info(f"{key}: {value}")

    # Load config JSON if specified. Use preprocessing configurations if available.
    preproc_configs = None
    if config.config_file_path is None:
        raise ValueError("Configuration file path must be specified.")
    with open(config.config_file_path) as file:
        config_dict = json.load(file)
        preproc_configs = config_dict.get('transforms', None)
        if config_dict.get('model', None) is None:
            raise ValueError('Model configuration not found in config file.')
    
    # Initialize client.
    client = Client(silence_logs=logging.ERROR)
    logger.info(f"Dashboard Link: {client.dashboard_link}")

    # NOTE: If reloading preprocessed data, we assume that the user already has saved preprocessors somewhere.
    # Handle reusing preprocessed data.
    if config.use_preprocessed:
        # TODO: Loading and saving.
        if config.preprocessed_data_path is None:
            raise ValueError("Preprocessed data path must be specified when reusing preprocessed data.")

        directory = os.path.dirname(config.preprocessed_data_path)
        dataset_name = os.path.splitext(os.path.basename(config.preprocessed_data_path))[0]
        data_glob_string = os.path.join(os.path.join(directory, f'{dataset_name}_*.parquet'))

        logger.info(f'Loading preprocessed data (parquet) from: {data_glob_string}')
        X_train = dd.read_parquet(data_glob_string)

        metadata_path = os.path.join(directory, f'{dataset_name}_metadata.pkl')

        logger.info(f'Loading information about preprocessed data from: {metadata_path}')
        with open(metadata_path, mode='rb') as file:
            metadata = pickle.load(file)
            num_continuous, num_categories = metadata
    else:
        # Check if input file exists.
        if not glob.glob(config.input_data_path):
            raise FileNotFoundError(f"Input data file not found: {config.input_data_path}")
        
        # Ensure that preprocessor directory exists so we can save.
        preprocessor_dir = os.path.dirname(config.preprocessor_path)
        os.makedirs(preprocessor_dir, exist_ok=True)
        logger.info(f"Created preprocessed data directory: {preprocessor_dir}")

        # Path (str) provided = want to save data, so ensure directories exist.
        if isinstance(config.preprocessed_data_path, str):
            preprocessed_dir = os.path.dirname(config.preprocessed_data_path)
            os.makedirs(preprocessed_dir, exist_ok=True)
            logger.info(f"Created preprocessor/metadata directory (for saving data transformers): {preprocessor_dir}")
        
        # No preprocessed data readily available: load raw training data and preprocess from scratch.
        train_df, original_continuous_columns, original_categorical_columns = read_large_data(
            input_file_paths=config.input_data_path, 
            blocksize=config.blocksize,
            categoricals=config.categoricals,
            excluded_cols=config.excluded_columns,
            file_format=config.file_format
        )
        logger.info(f"Continuous columns: {original_continuous_columns}")
        logger.info(f"Categorical columns: {original_categorical_columns}")
        logger.info(f'Number of Partitions: {train_df.npartitions}')

        logger.info("Dropping rows with NaN or infinite values...")
        train_df = train_df.replace([float('inf'), float('-inf')], float('nan')).dropna()
        (
            X_train,
            transforms,
            num_continuous,
            num_categories,
        ) = preprocess_large_data(
            train_df.copy(),
            original_continuous_columns,
            original_categorical_columns,
            pre_proc_method=config.pre_proc_method,
            pre_proc_config=preproc_configs
        )

        # Save preprocessed data to disk if desired.
        length = None
        try:
            if config.preprocessed_data_path is None:
                logger.warning('No preprocessed data path provided, skipping saving preprocessed data...')
            else:
                # TODO: Use regex so that this can accept glob strings?
                directory = os.path.dirname(config.preprocessed_data_path) # type: ignore
                dataset_name = os.path.splitext(os.path.basename(config.preprocessed_data_path))[0] # type: ignore
                name_gen = lambda i: f'{dataset_name}_{i}.parquet'

                metadata_path = os.path.join(directory, f'{dataset_name}_metadata.pkl')
                logger.info(f'Saving information about preprocessed data to: {metadata_path}')
                with open(metadata_path, mode='wb') as file:
                    metadata = (num_continuous, num_categories)
                    pickle.dump(metadata, file)

                logger.info(f'Saving preprocessed data to: {config.preprocessed_data_path}')
                save_task = X_train.to_parquet(directory, name_function=name_gen, compute=False)
                _, length = dask.compute(save_task, X_train.shape[0])
        except:
            logger.warning('Saving preprocessed data failed. Continuing without saving...')

        original_columns = list(train_df.columns)
        preprocessing_info = {
            'transforms': transforms,
            'reference_data': {
                'npartitions': X_train.npartitions,
                'original_cols': original_columns,
                'reordered_cols': list(X_train.columns),
                'nrows': length,
                'batch_size': config.batch_size
            } 
        }
        # Save preprocessors.
        logger.info(f'Saving preprocessors and related metadata at: {config.preprocessor_path}')
        with open(config.preprocessor_path, mode='wb') as file:
            pickle.dump(preprocessing_info, file)

    logger.info(f'Preprocessed data dimension: {len(X_train.columns)}')

    # TODO: Find way to save this together with state_dict in a nice way.
    # Either that or move this to some other place that's out of the way.
    directory = os.path.dirname(config.model_save_path)
    model_name = os.path.splitext(os.path.basename(config.model_save_path))[0]
    model_init_path = os.path.join(directory, f'{model_name}_init_params.pkl')
    model_configs = config_dict['model']

    logger.info(f'Saving model initialization parameters to: {model_init_path}')
    save_model_init_params(model_configs, X_train.shape[1], num_continuous, num_categories, model_init_path)

    # Train model (and save state_dict to provided path as side effect).
    model = train_model(
        config=config,
        X_train=X_train,
        num_conts=num_continuous,
        num_cats_per_col=num_categories
    )


if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Training')
    
    # Add configuration arguments
    parser = Config.add_arguments(parser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create config with parsed arguments
    config = Config.from_args(args)
    # Create model save directory if it doesn't exist
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    
    main(config)
