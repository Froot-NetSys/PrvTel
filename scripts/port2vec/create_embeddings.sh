#!/bin/bash

# Script used to train port2vec embeddings on Appraise dataset (ports and protocols), and then embed arbitrary data.

# Root of repo.
REPO="${HOME}/PrvTel"

# Data used to train port2vec embeddings (and the directory where the model will be saved).
PORT2VEC_MODEL_DIR="${REPO}/path/to/port2vec_model_dir"
PORT2VEC_TRAIN_DATA="${REPO}/path/to/appraise.csv"

# Can also specify a glob pattern like "path/to/your/target_data_*.csv" if not single file. Can choose between  .csv and .parquet files as output formats.
TARGET_DATA="path/to/your/target_data.csv"
EMBEDDED_DATA_PATH="path/to/embedded_output.parquet"

# The port_columns and protocol_columns indicate which column labels in the training data and target data correspond to ports and protocols.
# Note that the embedding_size and ann_trees specified need to match those used for training when loading a pretrained model. 
# The blocksize can be tuned based on your memory constraints (and whether you are using parquet or csv).
echo "Mapping ports and protocols to embeddings..."
python "${REPO}/scripts/embed_ports.py" \
    --input_data "${TARGET_DATA}" \
    --train_data "${PORT2VEC_TRAIN_DATA}" \
    --model_dir "${PORT2VEC_MODEL_DIR}" \
    --result_path "${EMBEDDED_DATA_PATH}" \
    --mode "transform" \
    --port_columns "L4_SRC_PORT" "L4_DST_PORT" \
    --protocol_columns "PROTOCOL" \
    --embedding_size 10 \
    --ann_trees 100 \
    --blocksize "1000MB" \
    --single_file \
    # --force_retrain