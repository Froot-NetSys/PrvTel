#!/bin/bash

# Training on Appraise dataset with port2vec embeddings (see create_embeddings.sh under scripts folder).

NAME="appraise_and_nfiot"

# Root of repo.
REPO="${HOME}/PrvTel"

# Add debug output for working directory after cd
echo "Working directory for training: $(pwd)"

# Provides the prefix t where the preprocessed data will be saved. If we have "path/to/data.parquet", we will get
# "path/to/data_i.parquet" where i is the partition number. We will also get "path/to/data_metadata.pkl" to
# store number of continuous features and number of categories per categorical.
PREPROCESSED_DATA_PATH="${REPO}/path/to/save_dir/${NAME}_preproc.parquet"
MODEL_SAVE_PATH="${REPO}/path/to/save_dir/${NAME}.pth"
PREPROCESSOR_PATH="${REPO}/path/to/save_dir/${NAME}_preprocessors.pkl"

# Regular training related parameters.
INPUT_DATA="${REPO}/path/to/word2vec_embedded_data/*.parquet"  # Update this to the actual path of your (word2vec) embedded data files.
RESULTS_DIR="${REPO}/results/${NAME}"
CONFIG_FILE_PATH="${REPO}/experiments/appraise_config.json" # Update this to the config for the corresponding data (or modify it as needed). Note that the config file is where you specify which features are continuous and which are categorical, so it's important to have this set correctly.
EXCLUDED_COLUMNS=("IPV4_SRC_ADDR" "IPV4_DST_ADDR")

# Create required directories
echo "Creating directory: ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

CUDA_VISIBLE_DEVICES="1,2" python "${REPO}/dist_train.py" \
    --input_data_path "${INPUT_DATA}" \
    --file_format "parquet" \
    --pre_proc_method "GMM_log" \
    --categoricals "ANOMALY" \
    --n_epochs 10 \
    --batch_size 2048 \
    --results_dir "${RESULTS_DIR}" \
    --blocksize "None" \
    --model_save_path "${MODEL_SAVE_PATH}" \
    --excluded_columns "${EXCLUDED_COLUMNS[@]}" \
    --preprocessor_path "${PREPROCESSOR_PATH}" \
    --config_file_path "${CONFIG_FILE_PATH}" \
    --num_chunks_cached 10 \
    --preprocessed_data_path "${PREPROCESSED_DATA_PATH}" \
    --use_preprocessed
