#!/bin/bash

# Inference on DrDoS dataset to generate synthetic data with saved model from train script.

NAME="drdos_udp"

REPO="${HOME}/PrvTel"

SYN_DATA="${REPO}/temp/${NAME}_temp.csv"

# Create required directories
echo "Creating directory: $(dirname "${SYN_DATA}")"
mkdir -p "$(dirname "${SYN_DATA}")"

# Add debug output to verify paths
echo "Current working directory: $(pwd)"
echo "Synthetic data path: ${SYN_DATA}"

# Add debug output for working directory after cd
echo "Working directory for inference: $(pwd)"

# Where the preprocessor and model from training are saved. Update these to the actual paths where you saved these during training.
PREPROCESSOR_PATH="${REPO}/path/to/save_dir/${NAME}_preprocessors.pkl"
MODEL_SAVE_PATH="${REPO}/path/to/save_dir/${NAME}.pth"

CUDA_VISIBLE_DEVICES="2" python "${REPO}/generate.py" \
    --syn_data_path "${SYN_DATA}" \
    --model_path "${MODEL_SAVE_PATH}" \
    --preprocessor_path "${PREPROCESSOR_PATH}" \
    --batch_size 8192  \
    --num_parts 4 \
    --single_file

# Getting embeddings. Training and transforming input data.
PORT2VEC_MODEL_DIR="${REPO}/path/to/port2vec_appraise"
PORT2VEC_TRAIN_DATA="${REPO}/path/to/appraise.csv"

OUT_DATA="${REPO}/data/syn/${NAME}/${NAME}_syn.csv"

# Reverse the embeddings.
echo "Reversing port2vec embeddings..."
python "${REPO}/scripts/embed_ports.py" \
    --input_data "${SYN_DATA}" \
    --train_data "${PORT2VEC_TRAIN_DATA}" \
    --model_dir "${PORT2VEC_MODEL_DIR}" \
    --result_path "${OUT_DATA}" \
    --mode "invert" \
    --port_columns "Source Port" "Destination Port" \
    --protocol_columns "Protocol" \
    --single_file