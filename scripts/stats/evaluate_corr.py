import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os
import datetime


def corr_heatmap_comparison(raw_corrs, syn_corrs):
    """
    Creates figure with correlation heatmaps of original (raw) and synthetic data.
    Takes correlation matrices corresponding to both datasets and returns a matplotlib Figure.
    """
    # Sort correlation matrices to ensure consistent feature order
    sorted_features = sorted(raw_corrs.index)
    raw_corrs = raw_corrs.loc[sorted_features, sorted_features]
    syn_corrs = syn_corrs.loc[sorted_features, sorted_features]

    # Only show lower triangle of the matrix.
    mask = np.triu(np.ones_like(raw_corrs, dtype=bool))
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 8), layout='constrained')

    # Create heatmaps.
    raw_ax = sns.heatmap(
        raw_corrs, 
        annot=True,  # Show correlation values
        cmap='coolwarm',  # Color scheme from red (negative) to blue (positive)
        center=0,  # Center the colormap at 0
        fmt='.2f',  # Show 2 decimal places
        square=True,
        ax=axs[0],
        vmin=-1,
        vmax=1,
        mask=mask
    )
    syn_ax = sns.heatmap(
        syn_corrs, 
        annot=True,  # Show correlation values
        cmap='coolwarm',  # Color scheme from red (negative) to blue (positive)
        center=0,  # Center the colormap at 0
        fmt='.2f',  # Show 2 decimal places
        square=True,
        ax=axs[1],
        vmin=-1,
        vmax=1,
        mask=mask
    )

    raw_ax.set_title('Original Data')

    syn_ax.set_title('Synthetic Data')

    # fig.subplots_adjust(top=0.8)  # Make more space at top
    fig.suptitle('Evaluating Synthetic Data Feature Correlations')

    return fig


def corr_heatmap_diffs(raw_corrs, syn_corrs):
    # Sort correlation matrices to ensure consistent feature order
    sorted_features = sorted(raw_corrs.index)
    raw_corrs = raw_corrs.loc[sorted_features, sorted_features]
    syn_corrs = syn_corrs.loc[sorted_features, sorted_features]
    
    diffs = (raw_corrs - syn_corrs).abs()
    fig, axs = plt.subplots(figsize=(10, 8), layout='constrained')

    # Only show lower triangle of the matrix.
    mask = np.triu(np.ones_like(diffs, dtype=bool))

    diffs_ax = sns.heatmap(
        diffs, 
        annot=True,  # Show correlation values
        cmap='coolwarm',  # Color scheme from red (negative) to blue (positive)
        center=0,  # Center the colormap at 0
        fmt='.2f',  # Show 2 decimal places
        square=True,
        ax=axs,
        vmin=0,
        vmax=2,
        mask=mask
    )

    diffs_mean = diffs.values.mean()
    diffs_std = diffs.values.std()


    # fig.subplots_adjust(top=0.8)  # Make more space at top
    fig.suptitle(f'Absolute Correlation Differences (Mean={diffs_mean:.2f}, Std={diffs_std:.2f})')

    return fig



def get_command_line_args():
    # Initialize
    parser = argparse.ArgumentParser(description='Evaluate Spearmans correlation between original and synthetic data.')

    # File related (inputs, saving output).
    parser.add_argument('--original', required=True, help='Path to original data CSV file')
    parser.add_argument('--synthetic', required=True, help='Path to synthetic data CSV file')
    parser.add_argument('--results_dir',default=None , help='Directory to save results (JSON format).')

    # Miscellaneous
    parser.add_argument('--excluded_columns', type=str, nargs='+', default=["time", "timestamp"], help='Labels to drop.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_command_line_args()

    # Read data
    original_df = pd.read_csv(args.original)
    synthetic_df = pd.read_csv(args.synthetic)

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

    raw_corrs = original_df.corr(method='spearman')
    syn_corrs = synthetic_df.corr(method='spearman')
    heatmaps_figure = corr_heatmap_comparison(raw_corrs, syn_corrs)

    heatmaps_figure.show()

    diffs_figure = corr_heatmap_diffs(raw_corrs, syn_corrs)

    diffs_figure.show()

    # Save heatmaps if possible.
    if args.results_dir is not None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = os.path.splitext(os.path.basename(args.synthetic))[0]

        heatmaps_file = os.path.join(args.results_dir, f'spearman_{dataset_name}_{timestamp}.png')
        diffs_file = os.path.join(args.results_dir, f'spearman_diffs_{dataset_name}_{timestamp}.png')

        print(f'Spearman correlation heatmaps saving to: {heatmaps_file}')
        heatmaps_figure.savefig(heatmaps_file)

        print(f'Correlation differences heatmap saving to: {diffs_file}')
        diffs_figure.savefig(diffs_file)