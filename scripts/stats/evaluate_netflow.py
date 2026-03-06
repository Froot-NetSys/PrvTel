# Summary: This script processes netflow data from a CSV file, summarizing OUT_BYTES, IN_BYTES, packet counts by protocol, and flow counts by destination port.

# Example usage: python evaluate_netflow.py path/to/your/netflow_data.csv --output path/to/output.json
# Example usage: 
# python ./scripts/evaluate_netflow.py ./data/raw/appraise_netflow_nine_features.csv

import pandas as pd
import argparse
import os
import json

def sum_out_bytes_by_port(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Group the DataFrame by 'L4_DST_PORT' and sum the 'OUT_BYTES' for each port
    result = df.groupby('L4_DST_PORT')['OUT_BYTES'].sum().reset_index()

    # Sort the results in descending order based on the summed 'OUT_BYTES'
    result = result.sort_values(by='OUT_BYTES', ascending=False)

    return result

def sum_in_bytes_by_port(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Group the DataFrame by 'L4_SRC_PORT' and sum the 'IN_BYTES' for each port
    result = df.groupby('L4_SRC_PORT')['IN_BYTES'].sum().reset_index()

    # Sort the results in descending order based on the summed 'IN_BYTES'
    result = result.sort_values(by='IN_BYTES', ascending=False)

    return result

def sum_packets_by_protocol(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Group the DataFrame by 'PROTOCOL' and count the packets for each protocol
    result = df.groupby('PROTOCOL').size().reset_index(name='PACKET_COUNT')

    # Sort the results in descending order based on the packet count
    result = result.sort_values(by='PACKET_COUNT', ascending=False)

    return result

def count_flows_by_dst_port(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Group the DataFrame by 'L4_DST_PORT' and count the number of flows for each port
    result = df['L4_DST_PORT'].value_counts().reset_index(name='FLOW_COUNT')
    result.columns = ['L4_DST_PORT', 'FLOW_COUNT']  # Rename columns for clarity

    # Sort the results in descending order based on the flow count
    result = result.sort_values(by='FLOW_COUNT', ascending=False)

    return result

def create_summary_json(output_out, output_in, output_protocol, output_flows, output_file, top_n=10):
    # Create a summary dictionary
    summary = {
        'top_out_bytes': output_out.head(top_n).to_dict(orient='records'),
        'top_in_bytes': output_in.head(top_n).to_dict(orient='records'),
        'top_protocols': output_protocol.head(top_n).to_dict(orient='records'),
        'top_flows': output_flows.head(top_n).to_dict(orient='records')
    }

    # Write the summary to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(summary, json_file, indent=4)

def save_to_csv(dataframe, filename):
    # Save the DataFrame to a CSV file
    dataframe.to_csv(filename, index=False)

if __name__ == "__main__":
    # Set up argument parsing to handle command-line input
    parser = argparse.ArgumentParser(description='Sum OUT_BYTES by L4_DST_PORT from a CSV file.')
    
    # Define the expected argument: the path to the input CSV file
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    
    # Add an optional argument to control output writing
    parser.add_argument('--no-output', action='store_true', help='Disable writing output files')
    
    # Add an optional argument for the number of top results
    parser.add_argument('--top-n', type=int, default=5, help='Number of top results to include in the summary')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call the function to sum OUT_BYTES by L4_DST_PORT using the provided CSV file
    output_out = sum_out_bytes_by_port(args.csv_file)
    
    # Call the function to sum IN_BYTES by L4_SRC_PORT using the provided CSV file
    output_in = sum_in_bytes_by_port(args.csv_file)
    
    # Call the function to sum packets by PROTOCOL using the provided CSV file
    output_protocol = sum_packets_by_protocol(args.csv_file)
    
    # Call the function to count flows by L4_DST_PORT using the provided CSV file
    output_flows = count_flows_by_dst_port(args.csv_file)
    
    # Create a results directory based on the input file name
    base_name = os.path.splitext(os.path.basename(args.csv_file))[0]  # Get the base name without extension
    results_dir = f"results/{base_name}"  # Define the results directory path
    os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Determine the output file paths
    output_file = os.path.join(results_dir, f"{base_name}_output.csv")  # Append _output.csv
    summary_output_file = os.path.join(results_dir, f"{base_name}_summary.json")  # Change extension to .json

    # Create the summary JSON file only if output writing is not disabled
    if not args.no_output:
        create_summary_json(output_out, output_in, output_protocol, output_flows, summary_output_file, top_n=args.top_n)
        print(f'Summary results saved to {summary_output_file}')

        # Save each result to its respective CSV file in the results directory
        out_bytes_filename = os.path.join(results_dir, "top_out_bytes.csv")
        in_bytes_filename = os.path.join(results_dir, "top_in_bytes.csv")
        protocol_counts_filename = os.path.join(results_dir, "protocol_counts.csv")
        flow_counts_filename = os.path.join(results_dir, "flow_counts.csv")

        # Save each result to its respective CSV file
        save_to_csv(output_out, out_bytes_filename)
        save_to_csv(output_in, in_bytes_filename)
        save_to_csv(output_protocol, protocol_counts_filename)
        save_to_csv(output_flows, flow_counts_filename)

        print(f'Results saved to {out_bytes_filename}, {in_bytes_filename}, {protocol_counts_filename}, and {flow_counts_filename}')
    else:
        print('Output writing is disabled.')
