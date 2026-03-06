#!/bin/bash

# Evaluating Count Sketch and Dyadic Count Sketch (DCS) on some large arbitrary dataset. Computes top k metrics with
# different tolerances, entropy, and max/mean quantile difference.

# Metrics will be written to a JSON under a results directory (same as in experiment scripts).
# Should be called from root directory of repo.

# COMMAND LINE ARGUMENTS

# File related (inputs, saving output).
# --input_data
#     Path to original data CSV file.
# --output_dir, default='results'
#     Directory to save JSON file of results to.

# # Miscellaneous
# --excluded_columns, default="time timestamp"
#     Labels to drop (if they are in the input).
# --blocksize, default='default'
#     Size of the chunks to split the data into.
# --categoricals, default="auto"
#     Columns to specify as categorical (or "auto" to infer them).
# --precision, default=None
#     Whether to round float data before using sketch. Can be used as a form of discretization (save computation).
#     Don't specify if you don't want to round the data.

# # Count sketch
# --num_columns, default=2000
#     Columns to initialize count sketch with.
# --num_rows, default=10
#     Rows to initialize count sketch with.
# --rho, default=2
#     Rho (random init) for both count sketch and DCS.

# DCS
# --universe', default=2**30
#     Universe size for dyadic count sketch (DCS) used for evaluating quantile error. 
#     Should be big enough to cover all unique values.
# --gamma', default=0.0325
#     Helps determine column count for DCS.

NAME="appraise_nfiot_sketch_eval"
REPO="${HOME}/PrvTel"
INPUT_DATA="path/to/NF_IoT.csv"  # Update this to the actual path of your data file.
RESULTS_DIR="results/${NAME}"

# Create required directories
echo "Creating directory: ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

# Add debug output to verify paths
echo "Current working directory: $(pwd)"
echo "Input data path: ${INPUT_DATA}"
echo "Results directory: ${RESULTS_DIR}"

echo "Evaluating sketch..."
python "${REPO}/scripts/evaluate_sketch.py" \
    --input_data "${INPUT_DATA}" \
    --excluded_columns "IPV4_SRC_ADDR" "IPV4_DST_ADDR" \
    --blocksize "100MB" \
    --output_dir "${RESULTS_DIR}" \
    --categoricals "Label" "L4_SRC_PORT" "L4_DST_PORT" "PROTOCOL" \
    # --precision 2

