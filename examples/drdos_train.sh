#!/bin/bash

# Training on DrDoS dataset to see if we can capture the temporal spike. Ports and protocols are mapped to word2vec embeddings (see create_embeddings.sh under scripts folder).

NAME="drdos_udp"

# Root of repo.
# Root of repo.
REPO="${HOME}/PrvTel"

# Add debug output for working directory after cd
echo "Working directory for training: $(pwd)"

# Provides the prefix to where the preprocessed data will be saved. If we have "path/to/data.parquet", we will get
# "path/to/data_i.parquet" where i is the partition number. We will also get "path/to/data_metadata.pkl" to
# store number of continuous features and number of categories per categorical.
PREPROCESSED_DATA_PATH="${REPO}/path/to/save_dir/${NAME}_preproc.parquet"
MODEL_SAVE_PATH="${REPO}/path/to/save_dir/${NAME}.pth"
PREPROCESSOR_PATH="${REPO}/path/to/save_dir/${NAME}_preprocessors.pkl"

# Regular training related parameters.
INPUT_DATA="${REPO}/path/to/word2vec_embedded_data/*.parquet"  # Update this to the actual path of your (word2vec) embedded data files.
RESULTS_DIR="${REPO}/results/${NAME}"
CONFIG_FILE_PATH="${REPO}/experiments/drdos_rnn_config.json" # Update this to the config for the corresponding data (or modify it as needed). Note that the config file is where you specify which features are continuous and which are categorical, so it's important to have this set correctly.
EXCLUDED_COLUMNS=(
    "Flow ID" "Source IP" "Destination IP" "Fwd Packet Length Max" "Fwd Packet Length Min" "Fwd Packet Length Mean" "Fwd Packet Length Std" "Bwd Packet Length Max" 
    "Bwd Packet Length Min" "Bwd Packet Length Mean" "Bwd Packet Length Std" "Flow IAT Mean" "Flow IAT Std" "Flow IAT Max" 
    "Flow IAT Min" "Fwd IAT Total" "Fwd IAT Mean" "Fwd IAT Std" "Fwd IAT Max" "Fwd IAT Min" "Bwd IAT Total" "Bwd IAT Mean" "Bwd IAT Std" "Bwd IAT Max" "Bwd IAT Min" 
    "Fwd PSH Flags" "Bwd PSH Flags" "Fwd URG Flags" "Bwd URG Flags" "Fwd Header Length" "Bwd Header Length" "Min Packet Length" 
    "Max Packet Length" "Packet Length Mean" "Packet Length Std" "Packet Length Variance" "FIN Flag Count" "SYN Flag Count" "RST Flag Count" "PSH Flag Count" 
    "ACK Flag Count" "URG Flag Count" "CWE Flag Count" "ECE Flag Count" "Down/Up Ratio" "Average Packet Size" "Avg Fwd Segment Size" "Avg Bwd Segment Size" 
    "Fwd Header Length.1" "Fwd Avg Bytes/Bulk" "Fwd Avg Packets/Bulk" "Fwd Avg Bulk Rate" "Bwd Avg Bytes/Bulk" "Bwd Avg Packets/Bulk" "Bwd Avg Bulk Rate" 
    "Subflow Fwd Packets" "Subflow Fwd Bytes" "Subflow Bwd Packets" "Subflow Bwd Bytes" "Init_Win_bytes_forward" "Init_Win_bytes_backward" "act_data_pkt_fwd" 
    "min_seg_size_forward" "Active Mean" "Active Std" "Active Max" "Active Min" "Idle Mean" "Idle Std" "Idle Max" "Idle Min" "Inbound" "Label" 
)

CUDA_VISIBLE_DEVICES="1,2" python "${REPO}/dist_train.py" \
    --input_data_path "${INPUT_DATA}" \
    --file_format "parquet" \
    --pre_proc_method "GMM_log" \
    --n_epochs 500 \
    --batch_size 1024 \
    --results_dir "${RESULTS_DIR}" \
    --blocksize "None" \
    --model_save_path "${MODEL_SAVE_PATH}" \
    --excluded_columns "${EXCLUDED_COLUMNS[@]}" \
    --preprocessor_path "${PREPROCESSOR_PATH}" \
    --config_file_path "${CONFIG_FILE_PATH}" \
    --num_chunks_cached 10 \
    --preprocessed_data_path "${PREPROCESSED_DATA_PATH}" \
    --use_preprocessed
