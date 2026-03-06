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
from contextlib import redirect_stdout


def iou(a, b):
    """Intersection over Union (IoU) of two 1D arrays.."""
    x = set(a)
    y = set(b)
    return len(x & y) / len(x | y)


def recall(a, b):
    """
    Calculates recall of b with respects to a. In other words, calculates
    the proportion of elements in a that are also included in b.
    """
    x = set(a)
    y = set(b)
    return len(x & y) / len(x)


def get_important_flows():
    pass


def top_src_with_most_unique_flows(raw, syn, k=5000):
    """
    Find top k source ports that hit the most unique destination ports.
    Returns a pd.Series, with the index being the ports and the values being the count of unique flows associated with it.
    """
    FLOW_COLUMNS = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL']

    # Get the counts for each flow.
    raw_flow_counts = raw.groupby(FLOW_COLUMNS).size()
    syn_flow_counts = syn.groupby(FLOW_COLUMNS).size()

    # Get the number of unique tuples for each source port.
    raw_num_dsts = raw_flow_counts.groupby('L4_SRC_PORT').size()
    syn_num_dsts = syn_flow_counts.groupby('L4_SRC_PORT').size()

    raw_num_dsts, syn_num_dsts = dask.compute(*(raw_num_dsts, syn_num_dsts))

    raw_top_srcs = raw_num_dsts.sort_values(ascending=False).iloc[:k]
    syn_top_srcs = syn_num_dsts.sort_values(ascending=False).iloc[:k]

    return raw_top_srcs, syn_top_srcs


def top_max_total_pkts(raw, syn, k=5000):
    """
    Find the top k destination ports with highest maximum volume of incoming and outgoing packets, where the destination port is within 0-1000.
    Returns a pd.Series, with the index being the ports and the values being the volume count.
    """
    raw = raw.copy()
    syn = syn.copy()

    raw['TOTAL_PKTS'] = raw['IN_PKTS'] + raw['OUT_PKTS']
    syn['TOTAL_PKTS'] = syn['IN_PKTS'] + syn['OUT_PKTS']

    # Filter for destination ports under 1000.
    raw = raw[raw['L4_DST_PORT'] <= 1000]
    syn = syn[syn['L4_DST_PORT'] <= 1000]

    FLOW_COLUMNS = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL']

    raw_max_totals = raw.groupby(FLOW_COLUMNS)['TOTAL_PKTS'].sum()
    syn_max_totals = syn.groupby(FLOW_COLUMNS)['TOTAL_PKTS'].sum()

    raw_max_totals, syn_max_totals = dask.compute(*(raw_max_totals, syn_max_totals))

    raw_top_totals = raw_max_totals.sort_values(ascending=False).iloc[:k].astype(int)
    syn_top_totals = syn_max_totals.sort_values(ascending=False).iloc[:k].astype(int)

    return raw_top_totals, syn_top_totals


def most_incoming_flows(raw, syn, k=5000):
    """
    Count the number of flows going to each destination port.
    Returns a pd.Series, with the index being the ports and the values being the flow count.
    """
    FLOW_COLUMNS = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL']

    # Get the counts for each flow.
    raw_flow_counts = raw.groupby(FLOW_COLUMNS).size()
    syn_flow_counts = syn.groupby(FLOW_COLUMNS).size()
    
    # Get the number of unique flows associated with each destination port.
    raw_flow_counts = raw_flow_counts.groupby('L4_DST_PORT').size()
    syn_flow_counts = syn_flow_counts.groupby('L4_DST_PORT').size()

    raw_flow_counts, syn_flow_counts = dask.compute(*(raw_flow_counts, syn_flow_counts))

    sorted_raw_flow_counts = raw_flow_counts.sort_values(ascending=False).iloc[:k].astype(int)
    sorted_syn_flow_counts = syn_flow_counts.sort_values(ascending=False).iloc[:k].astype(int)

    return sorted_raw_flow_counts, sorted_syn_flow_counts


def top_protocols(raw, syn, k=10):
    """
    Find the most frequent protocols for flows with low volume traffic (sub 1000 incoming/outgoing packets).
    Returns a pd.Series, with the index being the protocols and the values being the port count.
    """
    # Top protocols. For now, just filter for source/destination ports both under 1000.
    raw = raw.copy()
    syn = syn.copy()

    raw['TOTAL_PKTS'] = raw['IN_PKTS'] + raw['OUT_PKTS']
    syn['TOTAL_PKTS'] = syn['IN_PKTS'] + syn['OUT_PKTS']

    raw_mask = (raw['TOTAL_PKTS'] <= 1000)
    syn_mask = (syn['TOTAL_PKTS'] <= 1000)

    raw_protos = raw[raw_mask]['PROTOCOL'].value_counts(sort=True)
    syn_protos = syn[syn_mask]['PROTOCOL'].value_counts(sort=True)

    raw_protos, syn_protos = dask.compute(*(raw_protos, syn_protos))

    return raw_protos.iloc[:k].astype(int), syn_protos.iloc[:k].astype(int)


def top_flows_by_out_bytes(raw, syn, k=5000):
    """Find the top k flows with the most outgoing bytes (and the amount)."""
    OUT_FLOW_COLUMNS = ['L4_DST_PORT', 'PROTOCOL']

    # NOTE: I assume bytes from SRC_PORT -> DST_PORT correspond to "OUT_BYTES"...
    raw_in_bytes = raw.groupby(OUT_FLOW_COLUMNS)['OUT_BYTES'].sum()
    syn_in_bytes = syn.groupby(OUT_FLOW_COLUMNS)['OUT_BYTES'].sum()

    raw_in_bytes, syn_in_bytes = dask.compute(*(raw_in_bytes, syn_in_bytes))

    # Get ports with highest count. Coerce values to int so visual comparison is easier.
    raw_top_ports = raw_in_bytes.sort_values(ascending=False).iloc[:k].astype(int)
    syn_top_ports = syn_in_bytes.sort_values(ascending=False).iloc[:k].astype(int)

    return raw_top_ports, syn_top_ports


def read_data(raw_path, syn_path, excluded_cols=[], blocksize='default', precision=None):
    """Read original and synthetic data. Also do some basic data preparation."""
    raw_df = dd.read_csv(raw_path, blocksize=blocksize)
    syn_df = dd.read_csv(syn_path, blocksize=blocksize)

    # Round if necessary.
    if isinstance(precision, int):
        print(f'Rounding to decimal place: {args.precision}')
        raw_df = raw_df.round(args.precision)
        syn_df = syn_df.round(args.precision)

    # Only keep shared columns.
    shared_cols = list(set(raw_df.columns) & set(syn_df.columns))
    print(f'Shared columns: {shared_cols}')
    raw_df = raw_df[shared_cols]
    syn_df = syn_df[shared_cols]

    # Drop specified columns (if any).
    dropped_cols = set(shared_cols) & set(excluded_cols)
    print(f'Columns being dropped: {dropped_cols}')
    raw_df = raw_df.drop(columns=dropped_cols)
    syn_df = syn_df.drop(columns=dropped_cols)

    # Start loading into memory asynchronously.
    raw_df = raw_df.persist()
    syn_df = syn_df.persist()

    return raw_df, syn_df


def get_command_line_args():
    # Initialize
    parser = argparse.ArgumentParser(description='Evaluate on updated queries.')

    # File related (inputs, saving output).
    parser.add_argument('--original', required=True, help='Path to original data CSV file')
    parser.add_argument('--synthetic', required=True, help='Path to synthetic data CSV file')
    parser.add_argument('--results_dir',default=None , help='Directory to save results (JSON format).')
    parser.add_argument('--save_top_ports', default=False, action='store_true', 
                        help='Whether to save the top ports and protocol data from original and synthetic data.')

    # Miscellaneous
    parser.add_argument('--blocksize', default='default', help='Size of the chunks to split the data into.')
    parser.add_argument('--excluded_columns', type=str, nargs='+', default=["time", "timestamp"], help='Labels to drop.')
    parser.add_argument('--precision', type=int, default=None,
                          help='Whether to round float data before evaluation. Can be used as a form of discretization.')

    args = parser.parse_args()
    return args


def main(args):
    client = Client(silence_logs=logging.ERROR)
    print(f'Dashboard Link: {client.dashboard_link}')

    summary_stats = {}
    ports_and_protocols = {}
    
    # Read data
    raw_df, syn_df = read_data(
        args.original, 
        args.synthetic, 
        excluded_cols=args.excluded_columns,
        blocksize=args.blocksize,
        precision=args.precision
    )

    # Find most popular protocols per port.
    print(f'{SEPARATOR}\nTOP PROTOCOLS (IN_PKTS + OUT_PKTS <= 1000)\n{SEPARATOR}')

    raw_protos, syn_protos = top_protocols(raw_df, syn_df)

    proto_iou = iou(raw_protos.index, syn_protos.index)
    proto_recall = recall(raw_protos.index, syn_protos.index)

    ports_and_protocols['top_protocols'] = {
        'original': list(raw_protos.index),
        'synthetic': list(syn_protos.index)
    }
    summary_stats['top_protocols'] = {
        'iou': proto_iou,
        'recall': proto_recall
    }

    # Statistics on incoming bytes for given destination port.
    print(f'{SEPARATOR}\nTOP FLOWS RECEIVING MOST BYTES \n{SEPARATOR}')

    raw_top_flows, syn_top_flows = top_flows_by_out_bytes(raw_df, syn_df)

    flow_iou = iou(raw_top_flows.index, syn_top_flows.index)
    flow_recall = recall(raw_top_flows.index, syn_top_flows.index)

    # Restrict attention to "significant" ports (0-1000).
    raw_dst_ports = raw_top_flows.index.get_level_values('L4_DST_PORT')
    syn_dst_ports = syn_top_flows.index.get_level_values('L4_DST_PORT')

    raw_dst_ports_1000 = raw_dst_ports[raw_dst_ports <= 1000].sort_values()
    syn_dst_ports_1000 = syn_dst_ports[syn_dst_ports <= 1000].sort_values()

    raw_flows_1000 = raw_top_flows.sort_index().loc[(slice(raw_dst_ports_1000[0], raw_dst_ports_1000[-1]), slice(None))].index
    syn_flows_1000 = syn_top_flows.sort_index().loc[(slice(syn_dst_ports_1000[0], syn_dst_ports_1000[-1]), slice(None))].index

    flow_iou_1000 = iou(raw_flows_1000, syn_flows_1000)
    flow_recall_100 = recall(raw_flows_1000, syn_flows_1000)
    
    ports_and_protocols['top_flows_most_out_bytes'] = {
        'original': list(raw_top_flows.index),
        'synthetic': list(syn_top_flows.index),
        'original_1000': list(raw_flows_1000),
        'synthetic_1000': list(syn_flows_1000)
    }
    summary_stats['top_flows_most_out_bytes'] = {
        'iou': flow_iou,
        'recall': flow_recall,
        'iou_1000': flow_iou_1000,
        'recall_1000': flow_recall_100
    }

    # (DDoS) Finding the destination ports associated with the most number of flows (5-tuples).
    print(f'{SEPARATOR}\nNUM FLOWS FROM X -> DST PORT\n{SEPARATOR}')

    raw_most_flows, syn_most_flows = most_incoming_flows(raw_df, syn_df)
    port_iou = iou(raw_most_flows.index, syn_most_flows.index)
    port_recall = recall(raw_most_flows.index, syn_most_flows.index)

    # Restrict attention to "significant" ports (1-1000).
    raw_ports_1000 = raw_most_flows.index[raw_most_flows.index <= 1000]
    syn_ports_1000 = syn_most_flows.index[syn_most_flows.index <= 1000]

    port_iou_1000 = iou(raw_ports_1000, syn_ports_1000)
    port_recall_1000 = recall(raw_ports_1000, syn_ports_1000)

    ports_and_protocols['top_dst_ports_most_flows'] = {
        'original': list(raw_most_flows.index),
        'synthetic': list(syn_most_flows.index),
        'original_1000': list(raw_ports_1000),
        'synthetic_1000': list(syn_ports_1000)
    }
    summary_stats['top_dst_ports_most_flows'] = {
        'iou': port_iou,
        'recall': port_recall,
        'iou_1000': port_iou_1000,
        'recall_1000': port_recall_1000
    }

    # (DDoS) Destination ports with the highest volume of packets for a given record.
    print(f'{SEPARATOR}\nMOST TOTAL PACKETS FROM X -> DST PORT (0-1000)\n{SEPARATOR}')

    raw_max_pkts, syn_max_pkts = top_max_total_pkts(raw_df, syn_df)
    port_iou = iou(raw_max_pkts.index, syn_max_pkts.index)
    port_recall = recall(raw_max_pkts.index, syn_max_pkts.index)

    ports_and_protocols['top_flows_max_pkts'] = {
        'original': list(raw_max_pkts.index),
        'synthetic': list(syn_max_pkts.index),
    }
    summary_stats['top_flows_max_pkts'] = {
        'iou': port_iou,
        'recall': port_recall,
    }

    # (Super spreader) Finding source port that targets the most destination ports.
    print(f'{SEPARATOR}\nMOST SPREAD SRC PORTS\n{SEPARATOR}')

    raw_most_spread_srcs, syn_most_spread_srcs = top_src_with_most_unique_flows(raw_df, syn_df)

    src_port_iou = iou(raw_most_spread_srcs.index, syn_most_spread_srcs.index)
    src_port_recall = recall(raw_most_spread_srcs.index, syn_most_spread_srcs.index)

    # Restrict attention to "significant" ports (1-1000).
    raw_ports_1000 = raw_most_spread_srcs.index[raw_most_spread_srcs.index <= 1000]
    syn_ports_1000 = syn_most_spread_srcs.index[syn_most_spread_srcs.index <= 1000]

    port_iou_1000 = iou(raw_ports_1000, syn_ports_1000)
    port_recall_1000 = recall(raw_ports_1000, syn_ports_1000)

    ports_and_protocols['top_src_ports_most_unique_flows'] = {
        'original': list(raw_most_spread_srcs.index),
        'synthetic': list(syn_most_spread_srcs.index),
        'original_1000': list(raw_ports_1000),
        'synthetic_1000': list(syn_ports_1000)
    }
    summary_stats['top_src_ports_most_unique_flows'] = {
        'iou': src_port_iou,
        'recall': src_port_recall,
        'iou_1000': port_iou_1000,
        'recall_1000': port_recall_1000
    }

    return summary_stats, ports_and_protocols


if __name__ == '__main__':
    args = get_command_line_args()

    SEPARATOR = '=' * 50

    summary_stats, ports_and_protocols = main(args)

    print(summary_stats)
    # Save results to JSON file.
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_name = os.path.splitext(os.path.basename(args.synthetic))[0]
    os.makedirs(args.results_dir, exist_ok=True)

    output_file = os.path.join(args.results_dir, f'netflow_stats_{dataset_name}_{timestamp}.json')
    with open(output_file, 'w') as file:
        json.dump(summary_stats, file, indent=2)
    print(f'Saved results to: {output_file}')

    if args.save_top_ports:
        output_file = os.path.join(args.results_dir, f'netflow_ports_{dataset_name}_{timestamp}.json')
        with open(output_file, 'w') as file:
            json.dump(ports_and_protocols, file, indent=2)
        print(f'Saved results to: {output_file}')

    





    


