"""
Evaluate synthetic data quality by comparing it with original data.

This script calculates various metrics to assess how well synthetic data matches
the statistical properties of the original data, including:
- Single dimension metrics: entropy, top-k recall, quantile differences, mean quantile error
- Cross dimension metrics: cardinality error, L2 norm error, frequency error

Functions:
- read_data: Loads data from a CSV file, samples it, and excludes specified columns.
- calculate_entropy_difference: Computes the relative entropy difference between original and synthetic data.
- calculate_single_dim_metrics: Calculates single-dimensional metrics including entropy difference, top-k differences, KS test results, and mean quantile error.
- calculate_ks_test: Calculates the KS test for quantile comparison between original and synthetic data.
- calculate_cross_dim_metrics: Computes cross-dimensional metrics such as cardinality error, L2 norm error, and frequency error.
- calculate_top_k_difference: Evaluates top-K recall error between original and synthetic datasets.
- calculate_mean_quantile_error: Calculates the mean quantile error between original and synthetic data.

Example usage:
    python scripts/evaluate.py \
        --original data/original_data.csv \
        --synthetic data/synthetic_data.csv \
        --output-dir results

    python scripts/evaluate.py \
        --original e2e_system/data/toy_data/toy_syn_data.csv \
        --synthetic e2e_system/syn_data/toy_syn_data_output.csv \
        --output-dir results

The results will be saved as a JSON file in the output directory with format:
    eval_<dataset_name>_<timestamp>.json
"""

import scipy.stats
import pandas as pd
import dask.dataframe as dd
import numpy as np
from scipy.stats import ks_2samp
import itertools
from datetime import datetime
import os
import argparse
import json
import sys
import os
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot.compare_original_vs_synthetic_histograms import compare_original_vs_synthetic_histograms, compare_original_vs_synthetic_split_histograms
from scipy import stats

def read_data(input_data_filepath, sample_frac=1.0, excluded_cols=['time', 'timestamp']):
    # Load data into Dask DataFrame
    input_df = dd.read_csv(input_data_filepath)
    
    # Remove the 'time' column if it exists
    # Time columns often don't make sense to compare between original and synthetic data
    # because:
    # 1. Time values are usually unique, making entropy/frequency comparisons meaningless
    # 2. Synthetic data may use different time ranges or distributions
    # 3. Time-based patterns are better evaluated through specific time series metrics
    if isinstance(excluded_cols, str):
        excluded_cols = [col for col in excluded_cols.split()]
    if excluded_cols:
        dropped_cols = set(input_df.columns) & set(excluded_cols)
        input_df = input_df.drop(columns=dropped_cols)

    # Load a fraction to control memory usage.
    input_df = input_df.sample(frac=sample_frac).compute()

    # Find all column/features with categorical value
    original_categorical_columns = []
    categorical_len_count = 0
    for col in input_df:
        # Do not process the value
        if len(input_df[col].unique()) <= 10:
            original_categorical_columns.append(col)
            categorical_len_count += len(input_df[col].unique())

    # unused
    # original_continuous_columns = list(set(input_df.columns.values.tolist()) - set(original_categorical_columns))

    # Get common columns before returning
    return input_df, original_categorical_columns

def calculate_entropy_difference(original_data, synthetic_data):
    """Calculate the relative entropy difference between original and synthetic data."""
    # Get common columns
    common_columns = list(set(original_data.columns) & set(synthetic_data.columns))
    original_data = original_data[common_columns]
    synthetic_data = synthetic_data[common_columns]
    
    # Check for empty DataFrames
    if original_data.empty or synthetic_data.empty:
        return np.nan  # or handle as appropriate

    # Calculate entropy
    def get_entropy(df):
        # Calculate Shannon entropy for each column based on value frequencies
        # For continuous variables, consider discretizing or using a different method
        def calculate_entropy_for_column(column):
            if column.dtype == 'object' or len(column.unique()) <= 10:  # Categorical
                return scipy.stats.entropy(column.value_counts())
            else:  # Continuous
                discretized_column = np.histogram(column, bins='auto')[0]  # Discretize using histogram
                return scipy.stats.entropy(discretized_column / discretized_column.sum())  # Calculate entropy on discretized data
                # return calculate_entropy_for_continuous(column)
        
        return df.apply(calculate_entropy_for_column)
    
    # Get entropy for each column in original and synthetic data
    original_entropy = get_entropy(original_data)
    synthetic_entropy = get_entropy(synthetic_data)
    
    # Calculate relative entropy difference between original and synthetic data
    # Formula: |original - synthetic| / original
    # Lower values indicate better similarity between distributions
    entropy_difference = np.abs(original_entropy - synthetic_entropy)/original_entropy    

    return entropy_difference

def calculate_single_dim_metrics(original_data, synthetic_data, original_categorical_columns):
    # Get common columns
    common_columns = list(set(original_data.columns) & set(synthetic_data.columns))
    original_data = original_data[common_columns]
    synthetic_data = synthetic_data[common_columns]
    # Filter categorical columns to only include common columns
    original_categorical_columns = [col for col in original_categorical_columns if col in common_columns]
    
    # Calculate relative entropy difference using the new function
    entropy_difference = calculate_entropy_difference(original_data, synthetic_data)
    
    # Convert to DataFrame for consistent format with other metrics
    entropy_difference_df = pd.DataFrame(entropy_difference).transpose()
    entropy_difference_df.index = ['entropy_difference']

    # Calculate Top-K differences
    top_k_difference_K1000_tolerance_001_df = calculate_top_k_difference(original_data, synthetic_data, original_categorical_columns, K=1000, tolerance_pct=0.001)
    top_k_difference_K1000_tolerance_01_df = calculate_top_k_difference(original_data, synthetic_data, original_categorical_columns, K=1000, tolerance_pct=0.01)
    top_k_difference_K1000_tolerance_1_df = calculate_top_k_difference(original_data, synthetic_data, original_categorical_columns, K=1000, tolerance_pct=0.1)
    top_k_difference_K1000_tolerance_15_df = calculate_top_k_difference(original_data, synthetic_data, original_categorical_columns, K=1000, tolerance_pct=0.15)

    # New calculations for K=100
    top_k_difference_K100_tolerance_001_df = calculate_top_k_difference(original_data, synthetic_data, original_categorical_columns, K=100, tolerance_pct=0.001)
    top_k_difference_K100_tolerance_01_df = calculate_top_k_difference(original_data, synthetic_data, original_categorical_columns, K=100, tolerance_pct=0.01)
    top_k_difference_K100_tolerance_1_df = calculate_top_k_difference(original_data, synthetic_data, original_categorical_columns, K=100, tolerance_pct=0.1)
    top_k_difference_K100_tolerance_15_df = calculate_top_k_difference(original_data, synthetic_data, original_categorical_columns, K=100, tolerance_pct=0.15)

    # Calculate KS test for quantile comparison
    ks_results = calculate_ks_test(original_data, synthetic_data)

    # Calculate mean quantile error
    mean_quantile_error_df = calculate_mean_quantile_error(original_data, synthetic_data)
    
    # Combine all single dim metrics
    single_dim_result_df = pd.concat([
        ks_results, 
        mean_quantile_error_df, 
        top_k_difference_K1000_tolerance_001_df, 
        top_k_difference_K1000_tolerance_01_df, 
        top_k_difference_K1000_tolerance_1_df, 
        top_k_difference_K1000_tolerance_15_df,
        top_k_difference_K100_tolerance_001_df,
        top_k_difference_K100_tolerance_01_df,
        top_k_difference_K100_tolerance_1_df,
        top_k_difference_K100_tolerance_15_df,
        entropy_difference_df
    ])
    single_dim_result_df['mean'] = single_dim_result_df.mean(axis=1)
    single_dim_result_df['std'] = single_dim_result_df.std(axis=1)
    return single_dim_result_df.round(3)

def calculate_ks_test(original_data, synthetic_data):
    """Calculate the KS test for quantile comparison between original and synthetic data."""
    # Get common columns
    common_columns = list(set(original_data.columns) & set(synthetic_data.columns))
    original_data = original_data[common_columns]
    synthetic_data = synthetic_data[common_columns]
    
    ks_results = pd.Series(name='Max-quantile-difference')
    for column in common_columns:
        statistic, _ = ks_2samp(original_data[column], synthetic_data[column])
        ks_results[column] = statistic
    return ks_results.to_frame().T

def calculate_cross_dim_metrics(original_data, synthetic_data):
    # Fix: Get common columns between original and synthetic data
    common_columns = list(set(original_data.columns) & set(synthetic_data.columns))
    original_data = original_data[common_columns]
    synthetic_data = synthetic_data[common_columns]
    
    # Calculate 2D cardinality
    def calculate_2d_cardinality(df):
        combinations = list(itertools.combinations(df.columns, 2))
        cardinalities = {}
        for combo in combinations:
            cardinalities[combo] = df.groupby(list(combo)).size().reset_index().rename(columns={0:'count'}).shape[0]
        return cardinalities

    original_cardinality = calculate_2d_cardinality(original_data)
    syn_cardinality = calculate_2d_cardinality(synthetic_data)
    error_rate = {key: (abs(syn_cardinality[key] - original_cardinality[key]) / original_cardinality[key]) 
                 for key in original_cardinality.keys()}
    two_dim_cardinality_error_rate_df = pd.DataFrame.from_dict(error_rate, orient='index', columns=['cardinality_err_rate']).T

    # Calculate L2 norm error
    def calculate_l2_norm_error_2_dim(original_df, syn_df):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        column_combinations = list(itertools.combinations(original_df.columns, 2))
        error_dict = {}
        for column_comb in column_combinations:
            original_data_norm = scaler.fit_transform(original_df[list(column_comb)])
            syn_data_norm = scaler.transform(syn_df[list(column_comb)])
            l2_norm_original = np.linalg.norm(original_data_norm, 2)
            l2_norm_syn = np.linalg.norm(syn_data_norm, 2)
            l2_norm_error = abs(l2_norm_syn - l2_norm_original) / l2_norm_original
            error_dict[column_comb] = l2_norm_error
        return pd.DataFrame.from_dict(error_dict, orient='index', columns=['l2_norm_error_2d']).T

    # Calculate frequency error
    def calculate_frequency_error(original_df, syn_df):
        column_combinations = list(itertools.combinations(original_df.columns, 2))
        error_dict = {}
        for column_comb in column_combinations:
            original_freq = pd.Series(original_df[list(column_comb)].values.flatten()).value_counts(normalize=True)
            syn_freq = pd.Series(syn_df[list(column_comb)].values.flatten()).value_counts(normalize=True)
            all_index = original_freq.index.union(syn_freq.index)
            original_freq = original_freq.reindex(all_index, fill_value=0)
            syn_freq = syn_freq.reindex(all_index, fill_value=0)
            freq_error = np.abs(original_freq - syn_freq).sum() / 2
            error_dict[column_comb] = freq_error
        return pd.DataFrame.from_dict(error_dict, orient='index', columns=['freq_err_2d']).T

    l2_error_df = calculate_l2_norm_error_2_dim(original_data, synthetic_data)
    freq_error_df = calculate_frequency_error(original_data, synthetic_data)

    # Combine all cross dim metrics
    cross_dim_result_df = pd.concat([two_dim_cardinality_error_rate_df, l2_error_df, freq_error_df])
    cross_dim_result_df['mean'] = cross_dim_result_df.mean(axis=1)
    cross_dim_result_df['std'] = cross_dim_result_df.std(axis=1)
    return cross_dim_result_df.round(2)

def calculate_top_k_difference(original_data, synthetic_data, original_categorical_columns, K=1000, tolerance_pct=0.001):
    """Calculate top-K recall error between original and synthetic data.
    
    Args:
        original_data (pd.DataFrame): Original dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        original_categorical_columns (list): List of categorical column names
        K (int): Number of top values to compare (default: 1000)
        tolerance_pct (float): Tolerance percentage for continuous column comparison (default: 0.001 or 0.1%)
    
    Returns:
        pd.DataFrame: DataFrame containing top-K recall errors for each column
    """
    def get_top_k_values(df, k):
        return df.apply(lambda x: x.value_counts().index[:k].to_list())
    
    # Ensure we only process columns that exist in both dataframes
    common_columns = list(set(original_data.columns) & set(synthetic_data.columns))
    original_top_k = get_top_k_values(original_data[common_columns], K)
    synthetic_top_k = get_top_k_values(synthetic_data[common_columns], K)
    
    # Calculate recall scores for each column by comparing top-K values
    recall_scores = []
    for col in original_data.columns:
        # Get sets of top-K values for original and synthetic data
        a_set = set(original_top_k[col])
        b_set = set(synthetic_top_k[col])
        
        if col in original_categorical_columns:
            # For categorical columns: require exact matches between values
            intersection = a_set & b_set
        else:
            # For continuous columns: allow values within tolerance_pct range
            tolerance = tolerance_pct * (original_data[col].max() - original_data[col].min())
            intersection = {val for val in a_set if any(abs(val - syn_val) <= tolerance for syn_val in b_set)}
        
        # Calculate recall error as: 1 - (matching_values / total_values)
        recall_err = 1 - len(intersection) / len(a_set)
        recall_scores.append(recall_err)
    
    top_k_difference_df = pd.DataFrame([recall_scores], columns=common_columns)
    top_k_difference_df.index = [f'top_k_err_K{K}_tolerance{tolerance_pct}']
    
    return top_k_difference_df

def calculate_mean_quantile_error(original_data, synthetic_data):
    # Get common columns
    common_columns = list(set(original_data.columns) & set(synthetic_data.columns))
    original_data = original_data[common_columns]
    synthetic_data = synthetic_data[common_columns]
    
    mean_quantile_errors = {}
    for column in common_columns:
        # Basically ks2samp but with mean absolute difference.
        data1 = np.sort(original_data[column].values)
        data2 = np.sort(synthetic_data[column].values)
        n1 = data1.shape[0]
        n2 = data2.shape[0]

        data_all = np.concatenate([data1, data2])
        # using searchsorted solves equal data problem
        cdf1 = np.searchsorted(data1, data_all, side='right') / n1
        cdf2 = np.searchsorted(data2, data_all, side='right') / n2
        cddiffs = np.abs(cdf1 - cdf2)

        mean_error = cddiffs.mean()
        mean_quantile_errors[column] = mean_error

    return pd.Series(mean_quantile_errors, name='mean_quantile_error').to_frame().T

def main():
    parser = argparse.ArgumentParser(description='Evaluate synthetic data quality')
    parser.add_argument('--original', required=True, help='Path to original data CSV file')
    parser.add_argument('--synthetic', required=True, help='Path to synthetic data CSV file')
    parser.add_argument('--output-dir', default='results', help='Directory to save results')
    parser.add_argument('--sample-fraction', default=1.0, type=float, help='Percent of both data sets to sample.')
    parser.add_argument('--excluded-columns', type=str, nargs='+', default=["time", "timestamp"], help='Labels to drop.')
    parser.add_argument('--tolerance', default=0.001, type=float, 
                       help='Tolerance percentage for continuous column comparison (default: 0.001 or 0.1%%)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read data and remove time column from both datasets for consistent comparison
    original_data, categorical_columns = read_data(args.original, args.sample_fraction, args.excluded_columns)
    synthetic_data, _ = read_data(args.synthetic, args.sample_fraction, args.excluded_columns)
    
    # Get common columns and filter both datasets
    common_columns = list(set(original_data.columns) & set(synthetic_data.columns))
    # Sort common_columns to ensure consistent order
    common_columns.sort()
    
    original_data = original_data[common_columns]
    synthetic_data = synthetic_data[common_columns]
    categorical_columns = [col for col in categorical_columns if col in common_columns]
    
    print(f"Evaluating {len(common_columns)} common columns: {', '.join(common_columns)}")
    
    # Calculate metrics
    single_dim_results = calculate_single_dim_metrics(original_data, synthetic_data, categorical_columns)
    cross_dim_results = calculate_cross_dim_metrics(original_data, synthetic_data)

    # Create timestamp and dataset name for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_name = os.path.splitext(os.path.basename(args.synthetic))[0]
    
    # Convert DataFrames to dictionaries and handle tuple keys
    single_dict = single_dim_results.to_dict()
    cross_dict = {}
    for key, value in cross_dim_results.to_dict().items():
        if isinstance(key, tuple):
            # Convert tuple key to string representation
            cross_dict[str(key)] = value
        else:
            cross_dict[key] = value
    
    # Save results with metadata
    results = {
        'metadata': {
            'original_data': os.path.abspath(args.original),
            'synthetic_data': os.path.abspath(args.synthetic),
            'timestamp': timestamp,
            'num_rows': {
                'original': len(original_data),
                'synthetic': len(synthetic_data)
            },
            'columns': common_columns  # Use sorted common_columns
        },
        'single_dim_metrics': single_dict,
        'cross_dim_metrics': cross_dict
    }
    
    output_file = os.path.join(args.output_dir, f'eval_{dataset_name}_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Before final print statement, add histogram comparison
    # Generate histogram filename using same naming convention
    histogram_file = os.path.join(args.output_dir, f'distributions_{dataset_name}_{timestamp}.png')
    split_histogram_file = os.path.join(args.output_dir, f'scale_dist_{dataset_name}_{timestamp}.png')

    # Optional: Specify column order (categorical columns first, then continuous)
    column_order = categorical_columns + [col for col in common_columns if col not in categorical_columns]

    # Create comparison plot and save with specified column order
    compare_original_vs_synthetic_histograms(
        original_df=original_data,
        synthetic_df=synthetic_data,
        save_path=histogram_file,
        column_order=column_order
    )

    # For split histograms with same column order
    compare_original_vs_synthetic_split_histograms(
        original_df=original_data,
        synthetic_df=synthetic_data,
        save_path=split_histogram_file,
        column_order=column_order
    )
    
    # Summary of results
    print("Summary of Results:")
    print(f"Number of rows - Original: {len(original_data)}, Synthetic: {len(synthetic_data)}")
    
    print("Single-dimensional metrics:")
    mean_values = single_dict.get('mean')
    std_values = single_dict.get('std')
    print(f"{'Metric':<30} {'Mean':<15} {'Std':<15}")  # Header with aligned columns
    print("-" * 60)  # Separator line
    for key in mean_values.keys():
        print(f"{key:<30} {mean_values[key]:<15} {std_values[key]:<15}")

    print("Cross-dimensional metrics:")
    mean_values_cross = cross_dict.get('mean')
    std_values_cross = cross_dict.get('std')
    print(f"{'Metric':<30} {'Mean':<15} {'Std':<15}")  # Header with aligned columns
    print("-" * 60)  # Separator line
    for key in mean_values_cross.keys():
        print(f"{key:<30} {mean_values_cross[key]:<15} {std_values_cross[key]:<15}")

    print(f"Results saved to {output_file}")
    print(f"Distribution plots saved to {histogram_file}")
    print(f"Split distribution plots saved to {split_histogram_file}")

if __name__ == "__main__":
    main()

