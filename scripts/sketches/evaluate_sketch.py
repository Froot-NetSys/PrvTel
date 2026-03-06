from dask.distributed import Client
import dask.dataframe as dd
import pandas as pd
import numpy as np
import sys
import os
import argparse
import logging
import datetime
import json
from pathlib import Path

# Add the project root directory.
root_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(root_folder))

from sketch.sketch_utils import evaluate_sketch_topk_entropy, evaluate_sketch_quantile_error
from prvtel.data.preprocessing import read_large_data


def get_command_line_args():
    # Initialize
    parser = argparse.ArgumentParser(description='Evaluate effectiveness of sketch queries.')

    # File related (inputs, saving output).
    parser.add_argument('--input_data', required=True, help='Path to original data CSV file.')
    parser.add_argument('--output_dir', default='results', help='Directory to save results.')

    # Miscellaneous
    parser.add_argument('--excluded_columns', type=str, nargs='+', default=["time", "timestamp"], help='Labels to drop.')
    parser.add_argument('--blocksize', default='default', help='Size of the chunks to split the data into.')
    parser.add_argument('--categoricals', type=str, nargs='+', default=[],
                          help='Columns to specify as categorical.')
    parser.add_argument('--precision', type=int, default=None,
                          help='Whether to round float data before using sketch. Can be used as a form of discretization (save computation).')

    # Count sketch
    parser.add_argument('--num_columns', default=2000, help='Columns to initialize count sketch with.')
    parser.add_argument('--num_rows', default=10, help='Rows to initialize count sketch with.')
    parser.add_argument('-rho', default=2, help='Rho (random init) for both count sketch and DCS.')

    # DCS
    parser.add_argument('--universe', type=int, default=2**30, help='Universe size for DCS. Should be big enough to cover all unique values.')
    parser.add_argument('--gamma', type=float, default=0.0325, help='Helps determine column count for DCS.')

    args = parser.parse_args()
    return args


def save_results(results, args, timestamp=None):
    """
    Writes the aggregated comparison metrics to a JSON format.

    Args:
        results (pd.DataFrame): A DataFrame of the evaluation results. The index
            should be the name of the evaluation metric (e.g. 'mean-quantile-diff').
            The columns should be either a feature from the dataset or an aggregated
            measure (mean of all columns)
    """
    # Create timestamp and dataset name for filename
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    dataset_name = os.path.splitext(os.path.basename(args.input_data))[0]
    
    # Convert DataFrames to dictionaries and handle tuple keys
    metrics_dict = results.to_dict()
    
    # Save results with metadata
    results = {
        'metadata': {
            'input_data': os.path.abspath(args.input_data),
            'timestamp': timestamp,
            'columns': list(results.columns)
        },
        'metrics': metrics_dict
    }
    
    output_file = os.path.join(args.output_dir, f'sketch_eval_{dataset_name}_{timestamp}.json')

    print(f'Saving JSON of results to {output_file}')

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    client = Client(silence_logs=logging.ERROR)
    print(f'Dashboard Link: {client.dashboard_link}')

    args = get_command_line_args()

    # Normalize numeric columns to float to avoid metadata mismatch issues.
    # TODO: Is there a better way to do this?
    infer_types = dd.read_csv(args.input_data, blocksize=args.blocksize)
    col_types = infer_types.dtypes.to_dict()
    for col, dtype in col_types.items():
        if dtype == np.int64:
            col_types[col] = 'float64'

    ddf, cont_cols, cat_cols = read_large_data(
        input_file_paths=args.input_data,
        excluded_cols=args.excluded_columns,
        categoricals=args.categoricals,
        blocksize=args.blocksize,
        dtype=col_types
    )
    # Clean data.
    ddf = ddf.replace([np.inf, -np.inf], np.nan).dropna()

    if args.precision is not None:
        ddf = ddf.round(args.precision)
    
    # NOTE: Data must fit in memory to persist, but this should be fine for most
    # relevant baselines to evaluate...
    ddf = ddf.persist()

    print(f'Continuous Columns: {cont_cols}')
    print(f'Categorical Columns: {cat_cols}')

    results = []
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Evaluate these together since they use regular count sketch.
    topk_entropy_results = evaluate_sketch_topk_entropy(
        ddf,
        cont_cols=cont_cols,
        cols=args.num_columns, 
        rows=args.num_rows, 
        rho=args.rho
    )

    # Save results so far.
    results = [topk_entropy_results] + results
    results_df = pd.concat(results)

    save_results(results_df, args, timestamp=timestamp)

    # These require DCS, so separate computation.
    quantile_results = evaluate_sketch_quantile_error(
        ddf,
        universe=args.universe,
        gamma=args.gamma,
        rho=args.rho 
    )

    # Save results so far. Adding to the front of list to
    # keep order of results consistent with evaluate.py
    results = [quantile_results] + results
    results_df = pd.concat(results)

    save_results(results_df, args, timestamp=timestamp)