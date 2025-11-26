#!/bin/bash

# Get the directory where the script is located (repo root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Define directory paths
DATA_DIR="${SCRIPT_DIR}/data"
UTILS_DIR="${SCRIPT_DIR}/utils"

mkdir -p "${DATA_DIR}/sing_label_data2_1"

if [[ ! -f "${UTILS_DIR}/sing_label_method.py" ]]; then
    echo "ERROR: ${UTILS_DIR}/sing_label_method.py not found"
    exit 1
fi

python3 "${UTILS_DIR}/sing_label_method.py" \
  --train_embeddings "data/resampled_data/multiclass_ADASYN_embeddings.npy" \
  --train_labels "data/resampled_data/multiclass_ADASYN_labels.npy" \
  --output "data/sing_label_data2_1/multiclass_ADASYN_single_label.npy"

python3 "${UTILS_DIR}/sing_label_method.py" \
  --train_embeddings "data/resampled_data/multiclass_CondensedNearestNeighbour_embeddings.npy" \
  --train_labels "data/resampled_data/multiclass_CondensedNearestNeighbour_labels.npy" \
  --output "data/sing_label_data2_1/multiclass_CondensedNearestNeighbour_single_label.npy"

python3 "${UTILS_DIR}/sing_label_method.py" \
  --train_embeddings "data/resampled_data/multiclass_RandomOverSampler_embeddings.npy" \
  --train_labels "data/resampled_data/multiclass_RandomOverSampler_labels.npy" \
  --output "data/sing_label_data2_1/multiclass_RandomOverSampler_single_label.npy"

python3 "${UTILS_DIR}/sing_label_method.py" \
  --train_embeddings "data/resampled_data/multiclass_RandomUnderSampler_embeddings.npy" \
  --train_labels "data/resampled_data/multiclass_RandomUnderSampler_labels.npy" \
  --output "data/sing_label_data2_1/multiclass_RandomUnderSampler_single_label.npy"

python3 "${UTILS_DIR}/sing_label_method.py" \
  --train_embeddings "data/resampled_data/multiclass_SMOTE_embeddings.npy" \
  --train_labels "data/resampled_data/multiclass_SMOTE_labels.npy" \
  --output "data/sing_label_data2_1/multiclass_SMOTE_single_label.npy"

python3 "${UTILS_DIR}/sing_label_method.py" \
  --train_embeddings "data/resampled_data/multiclass_SMOTEENN_embeddings.npy" \
  --train_labels "data/resampled_data/multiclass_SMOTEENN_labels.npy" \
  --output "data/sing_label_data2_1/multiclass_SMOTEENN_single_label.npy"



