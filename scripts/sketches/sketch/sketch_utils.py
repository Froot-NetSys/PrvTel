from .count_sketch import create_count_sketches, create_dcs
import math
import scipy as sp
import numpy as np
import pandas as pd
import dask
import itertools


def discretize_counts(val_counts):
    """
    Takes a pd.Series of the form returned by pd.value_counts.
    Bins the values represented by the value counts and returns
    a numpy array with the normalized counts for each bin.
    """
    # Get unique values (and their counts) in ascending order.
    freqs = val_counts.sort_index()

    uniques = freqs.index.values
    counts = freqs.values

    # Calculating number of bins in the same manner as np.histograms with bins='auto'.
    n = counts.sum()
    iqr = sp.stats.iqr(uniques)
    r = uniques[-1] - uniques[0] # The range.

    # Sturges and Freedman Diaconis (FD) Estimator
    sturges = math.log(n, 2) + 1
    fd = 2 * iqr / n ** (1 / 3)

    bin_width = min(sturges, fd)
    num_bins = math.ceil(r / bin_width)

    discretized = np.histogram(uniques, range=(uniques[0], uniques[-1]), bins=num_bins, weights=counts)[0] / n

    return discretized


def get_topk_entropy(ddf, k, discretized_cols=[]):
    """
    Fused computation of actual top k and entropy values on Dask DataFrame.
    Doing this to avoid repeated computations of value counts.
    """
    # Compute value counts in parallel on each column (if possible).
    jobs = [ddf[col].value_counts() for col in ddf.columns]
    val_counts = dask.compute(*jobs)

    # Compute metrics.
    # TODO: Discretize continuous variables?
    topk = [values.index[:k].tolist() for values in val_counts]

    entropy_values = []
    for col, vals in zip(ddf.columns, val_counts):
        # Discretize (bin) continuous data.
        if col in discretized_cols:
            values = discretize_counts(vals)
            entropy_values.append(values)
        else:
            entropy_values.append(vals)

    entropies = [sp.stats.entropy(values) for values in entropy_values]

    return topk, entropies


@dask.delayed
def estimate_topk(cs, unique_values, k):
    """Get top k of a column as determined by sketch."""
    # TODO: Original author didn't use abs here, but just in case...
    ans = np.abs(cs.batch_query(unique_values))
    val_freqs = sorted(zip(unique_values, ans), key=lambda x: x[1], reverse=True)[:k]
    return [val for val, _ in val_freqs]


def compare_topk(
    real_topk, 
    sketch_topk, 
    k_list, 
    tolerances, 
    ddf, 
    cont_cols
):
    # Calculate recall scores for each column by comparing top-K values
    columns = list(ddf.columns)
    k_tolerance_pairs = itertools.product(k_list, tolerances)
    
    ranges = {col: (ddf[col].max() - ddf[col].min()).persist() for col in columns}

    def helper(k, tolerance_pct):
        recall_scores = []
        for a, b, col in zip(real_topk, sketch_topk, columns):
            # Get sets of top-K values for original and synthetic data
            a_set = set(a[:k])
            b_set = set(b[:k])
            
            if col not in cont_cols:
                # For categorical columns: require exact matches between values
                intersection = a_set & b_set
            else:
                # For continuous columns: allow values within tolerance_pct range
                tolerance = tolerance_pct * ranges[col].compute()
                intersection = {val for val in a_set if any(abs(val - syn_val) <= tolerance for syn_val in b_set)}
            
            # Calculate recall error as: 1 - (matching_values / total_values)
            recall_err = 1 - len(intersection) / len(a_set)
            recall_scores.append(recall_err)

        top_k_difference_df = pd.DataFrame([recall_scores], columns=columns)
        top_k_difference_df.index = [f'top_k_err_K{k}_tolerance{tolerance_pct}']
        
        return top_k_difference_df
    
    top_k_dfs = []
    for k, tol in k_tolerance_pairs:
        df = helper(k, tol)
        top_k_dfs.append(df)
    result = pd.concat(top_k_dfs)

    return result


@dask.delayed
def estimate_entropy(cs, unique_values, discretize=False):
    """Get entropy of a column via sketch."""
    # TODO: Sketch producing negative values, which makes entropy calculation wonky...
    # Not sure if this makes more sense than clipping at 0.
    ans = np.abs(cs.batch_query(unique_values))
    if discretize:
        val_counts = pd.Series(ans, index=unique_values)
        ans = discretize_counts(val_counts)
    return sp.stats.entropy(ans)


def compare_entropies(real_entropies, sketch_entropies, ddf):
    diffs = []
    for real, sketch in zip(real_entropies, sketch_entropies):
        diffs.append(abs(real - sketch) / real)

    entropy_difference_df = pd.DataFrame([diffs], columns=list(ddf.columns))
    entropy_difference_df.index = ['entropy_difference']

    return entropy_difference_df


def compare_quantiles(rank_map, dcs_map, ddf):
    quantile_jobs = [find_quantile_diffs(dcs_map[col], rank_map[col]) for col in ddf.columns]
    sketch_quantile_errors = dask.compute(*quantile_jobs)

    max_errors = [max_error for (max_error, _) in sketch_quantile_errors]
    mean_errors = [mean_error for (_, mean_error) in sketch_quantile_errors]

    quantile_errors_df = pd.DataFrame([max_errors, mean_errors], columns=list(ddf.columns))
    quantile_errors_df.index = ['max-quantile-difference', 'mean-quantile-difference']

    return quantile_errors_df


@dask.delayed
def find_quantile_diffs(dcs, cum_counts):
    """
    Takes a pd.Series, where index contains unique items (sorted) and the values
    are the cumulative frequency (left to right). Compares those with the sketch
    estimates.
    """
    # make sure data is a dictionary of item:rank
    max_rank_diff = 0
    N = cum_counts.iat[-1] # The number of data points in the original.
    true_ranks = cum_counts

    # Convert to Series for queries.
    query_items = pd.Series((true_ranks.index + 1).values)
    estimate_ranks = dcs.batch_query(query_items)

    diffs = (estimate_ranks - true_ranks).abs()

    max_rank_diff = max(max_rank_diff, diffs.max())
    mean_rank_diff = diffs.mean()

    normalized_max_diff = max_rank_diff / N
    normalized_mean_diff = mean_rank_diff / N

    return normalized_max_diff, normalized_mean_diff


def evaluate_sketch_topk_entropy(ddf, cont_cols, tolerances=[0.001, 0.01, 0.1, 0.15], k_list=[100, 1000], cols=2000, rows=10, rho=2):
    """Same as notebook. Evaluate top k and entropy for sketch."""

    # Find the top k of the largest k to avoid repeated computations.
    K = max(k_list)
    columns = list(ddf.columns)

    col2CS = create_count_sketches(ddf, cols=cols, rows=rows, rho=rho)

    # Populate col2Values, the unique elements for each column.
    uniques = [ddf[col].unique() for col in columns]
    results = dask.compute(*uniques)
    col2Values = {col: result for col, result in zip(columns, results)}

    # Get real top k and entropy.
    top_k, entropies = get_topk_entropy(ddf, K, discretized_cols=cont_cols)

    # Get sketch estimated top k.
    sketch_jobs = [estimate_topk(cs, col2Values[col], K) for col, cs in col2CS.items()]
    sketch_top_k = dask.compute(*sketch_jobs)

    # Calculate recall scores for each column by comparing top-K values
    top_k_results = compare_topk(
        real_topk=top_k,
        sketch_topk=sketch_top_k,
        k_list=k_list,
        tolerances=tolerances,
        ddf=ddf,
        cont_cols=cont_cols
    )

    # Calculate sketch based entropy, discretizing if features are continuous.
    entropy_jobs = [estimate_entropy(cs, col2Values[col], discretize=(col in cont_cols)) for col, cs in col2CS.items()]
    sketch_entropies = dask.compute(*entropy_jobs)

    entropy_results = compare_entropies(entropies, sketch_entropies, ddf)

    results = pd.concat([top_k_results, entropy_results])
    results['mean'] = results.mean(axis=1)
    results['std'] = results.std(axis=1)
    results = results.round(3)

    return results


def evaluate_sketch_quantile_error(ddf, universe=2**30, gamma=0.0325, rho=2):
    """Evaluate quantile error using dyadic count sketch."""
    columns = list(ddf.columns)

    # Compute the DCS.
    col2DCS = create_dcs(ddf, universe=universe, gamma=gamma, rho=rho)

    # Get value counts for each column
    val_count_jobs = [ddf[col].value_counts() for col in columns]
    results = dask.compute(*val_count_jobs)
    val_counts = {col: result for col, result in zip(columns, results)}

    # Get col2Ranks, a cumulative count of unique elts in ascending order?
    col2Ranks = {}
    for col in columns:
        entry = val_counts[col].sort_index().cumsum()
        col2Ranks[col] = entry

    # Evaluate sketch estimates against each column.
    results = compare_quantiles(col2Ranks, col2DCS, ddf)
    results['mean'] = results.mean(axis=1)
    results['std'] = results.std(axis=1)
    results = results.round(3)

    return results
