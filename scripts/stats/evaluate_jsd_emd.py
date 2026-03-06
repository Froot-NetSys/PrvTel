from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
import dask
import argparse
import logging
from pprint import pprint
import json
import datetime
import os
import pandas as pd


def jsd(raw, syn):
    """Calculates Jensen-Shannon distance between features of two DataFrames."""
    if not raw.columns.equals(syn.columns):
        raise ValueError('Inputs should have same features.')
    
    # Calculate frequencies for each column.
    all_raw_freqs = dask.compute(*[raw[col].value_counts() for col in raw.columns])
    all_syn_freqs = dask.compute(*[syn[col].value_counts() for col in syn.columns])

    # Collect frequencies together in one place.
    frequencies = {col: (raw_freqs, syn_freqs) for col, raw_freqs, syn_freqs in zip(raw.columns, all_raw_freqs, all_syn_freqs)}

    jsd_jobs = []
    for col, freqs in frequencies.items():
        raw_freqs, syn_freqs = freqs
        # NOTE: Need same length arrays for Jensen-Shannon, so pad missing values with 0.
        # Ex: Raw data only has freqs for elements 1, 2, and 4, while the synthetic data
        # has them for 1, 2, and 3. Then raw has freq 0 for element 3, and syn has freq 0 for element 4.
        padded_freqs = pd.concat([raw_freqs[~raw_freqs.index.duplicated()], syn_freqs[~syn_freqs.index.duplicated()]], axis=1, join='outer')
        padded_freqs = padded_freqs.fillna(0)

        raw_freqs = padded_freqs.iloc[:, 0]
        syn_freqs = padded_freqs.iloc[:, 1]

        jsd_job = dask.delayed(jensenshannon)(raw_freqs.values, syn_freqs.values)
        jsd_jobs.append(jsd_job)

    js_dists = dask.compute(*jsd_jobs)

    results = {col: js_dist for col, js_dist in zip(raw.columns, js_dists)}

    return results


def emd(raw, syn):
    """Calculates Earth Mover's (Wasserstein-1) distance between features of two DataFrames."""
    if not raw.columns.equals(syn.columns):
        raise ValueError('Inputs should have same features.')
    
    # Calculate frequencies for each column.
    all_raw_freqs = dask.compute(*[raw[col].value_counts() for col in raw.columns])
    all_syn_freqs = dask.compute(*[syn[col].value_counts() for col in syn.columns])

    # Collect frequencies together in one place.
    frequencies = {col: (raw_freqs, syn_freqs) for col, raw_freqs, syn_freqs in zip(raw.columns, all_raw_freqs, all_syn_freqs)}

    emd_jobs = []
    for col, freqs in frequencies.items():
        raw_freqs, syn_freqs = freqs

        emd_job = dask.delayed(wasserstein_distance)(
            raw_freqs.index.values,
            syn_freqs.index.values,
            raw_freqs.values,
            syn_freqs.values
        )
        emd_jobs.append(emd_job)

    em_dists = dask.compute(*emd_jobs)

    results = {col: em_dist for col, em_dist in zip(raw.columns, em_dists)}

    return results


def get_command_line_args():
    # Initialize
    parser = argparse.ArgumentParser(description='Evaluate synthetic data quality with Jensen-Shannon distance (categorical features) '
                                     'or Earth Mover\'s distance (continuous features).')

    # File related (inputs, saving output).
    parser.add_argument('--original', required=True, help='Path to original data CSV file')
    parser.add_argument('--synthetic', required=True, help='Path to synthetic data CSV file')
    parser.add_argument('--results_dir',default=None , help='Directory to save results (JSON format).')

    # Miscellaneous
    parser.add_argument('--blocksize', default='default', help='Size of the chunks to split the data into.')
    parser.add_argument('--excluded_columns', type=str, nargs='+', default=["time", "timestamp"], help='Labels to drop.')
    parser.add_argument('--categoricals', type=str, nargs='+', default=[],
                          help='Space-separated list of categorical columns. Example: --categoricals col1 col2')
    parser.add_argument('--precision', type=int, default=None,
                          help='Whether to round float data before evaluation. Can be used as a form of discretization.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_command_line_args()

    client = Client(silence_logs=logging.ERROR)
    print(f'Dashboard Link: {client.dashboard_link}')
    
    # Read data
    original_df = dd.read_csv(args.original, blocksize=args.blocksize)
    synthetic_df = dd.read_csv(args.synthetic, blocksize=args.blocksize)

    # Round if necessary.
    if isinstance(args.precision, int):
        print(f'Rounding to decimal place: {args.precision}')
        original_df = original_df.round(args.precision)
        synthetic_df = synthetic_df.round(args.precision)

    # Only keep shared columns.
    shared_cols = list(set(original_df.columns) & set(synthetic_df.columns))
    print(f'Shared columns: {shared_cols}')
    original_df = original_df[shared_cols]
    synthetic_df = synthetic_df[shared_cols]

    # Drop specified columns (if any).
    dropped_cols = set(shared_cols) & set(args.excluded_columns)
    print(f'Columns being dropped: {dropped_cols}')
    original_df = original_df.drop(columns=dropped_cols)
    synthetic_df = synthetic_df.drop(columns=dropped_cols)

    # Start loading into memory asynchronously.
    original_df = original_df.persist()
    synthetic_df = synthetic_df.persist()

    cat_cols = list(set(args.categoricals) & set(original_df.columns))
    cont_cols = list(set(original_df.columns) - set(cat_cols))
    print(f'Continuous features: {cont_cols}')
    print(f'Categorical features: {cat_cols}')

    # Compute distance metrics.
    results = {}
    if cat_cols:
        js_dists = jsd(original_df[cat_cols], synthetic_df[cat_cols])
        results['Jensen-Shannon'] = js_dists
    if cont_cols:
        em_dists = emd(original_df[cont_cols], synthetic_df[cont_cols])
        results['Earth Mover\'s'] = em_dists

    pprint(results)

    # Save results if possible.
    if args.results_dir is not None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = os.path.splitext(os.path.basename(args.synthetic))[0]
        results_file = os.path.join(args.results_dir, f'jsd_emd_{dataset_name}_{timestamp}.json')
        print(f'Distance metrics saving to: {results_file}')
        with open(results_file, mode='w') as f:
            json.dump(results, f, indent=2)

