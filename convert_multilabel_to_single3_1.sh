#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

DATA_DIR="${SCRIPT_DIR}/data"
UTILS_DIR="${SCRIPT_DIR}/utils"

mkdir -p "${DATA_DIR}/sing_label_data3_1"

if [[ ! -f "${UTILS_DIR}/sing_label_method.py" ]]; then
    echo "ERROR: sing_label_method.py not found"
    exit 1
fi

echo "Converting 3:1 multilabel → single-label..."

python3 "${UTILS_DIR}/sing_label_method.py" \
  --train_embeddings "data/resampled_data_multiclass31/multiclass_ADASYN_embeddings.npy" \
  --train_labels "data/resampled_data_multiclass31/multiclass_ADASYN_labels.npy" \
  --output "data/sing_label_data3_1/multiclass_ADASYN_single_label.npy"

python3 "${UTILS_DIR}/sing_label_method.py" \
  --train_embeddings "data/resampled_data_multiclass31/multiclass_RandomOverSampler_embeddings.npy" \
  --train_labels "data/resampled_data_multiclass31/multiclass_RandomOverSampler_labels.npy" \
  --output "data/sing_label_data3_1/multiclass_RandomOverSampler_single_label.npy"

python3 "${UTILS_DIR}/sing_label_method.py" \
  --train_embeddings "data/resampled_data_multiclass31/multiclass_RandomUnderSampler_embeddings.npy" \
  --train_labels "data/resampled_data_multiclass31/multiclass_RandomUnderSampler_labels.npy" \
  --output "data/sing_label_data3_1/multiclass_RandomUnderSampler_single_label.npy"

python3 "${UTILS_DIR}/sing_label_method.py" \
  --train_embeddings "data/resampled_data_multiclass31/multiclass_SMOTE_embeddings.npy" \
  --train_labels "data/resampled_data_multiclass31/multiclass_SMOTE_labels.npy" \
  --output "data/sing_label_data3_1/multiclass_SMOTE_single_label.npy"

python3 "${UTILS_DIR}/sing_label_method.py" \
  --train_embeddings "data/resampled_data_multiclass31/multiclass_SMOTEENN_embeddings.npy" \
  --train_labels "data/resampled_data_multiclass31/multiclass_SMOTEENN_labels.npy" \
  --output "data/sing_label_data3_1/multiclass_SMOTEENN_single_label.npy"

echo "3:1 conversion complete."

