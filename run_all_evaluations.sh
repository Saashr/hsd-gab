#!/bin/bash

# Get the directory where the script is located (repo root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Define directory paths
DATA_DIR="${SCRIPT_DIR}/data"
GOLD_LABELS_DIR="${SCRIPT_DIR}/gold_labels"
UTILS_DIR="${SCRIPT_DIR}/utils"

TEST_EMB="${GOLD_LABELS_DIR}/test_embeddings.npy"
TEST_LABELS="${GOLD_LABELS_DIR}/test_labels.npy"

# Check test files exist
if [[ ! -f "$TEST_EMB" || ! -f "$TEST_LABELS" ]]; then
  echo "ERROR: Required test files not found: $TEST_EMB or $TEST_LABELS"
  exit 1
fi

for DIR in sing_label_data sing_label_data2_1 sing_label_data3_1; do
  DIR_PATH="${DATA_DIR}/${DIR}"
  echo "== Directory: $DIR_PATH =="
  if [[ ! -d "$DIR_PATH" ]]; then
    echo "WARNING: Directory $DIR_PATH not found, skipping..."
    continue
  fi
  for FILE in "${DIR_PATH}"/*.npy; do
    if [[ ! -f "$FILE" ]]; then
      echo "WARNING: Training file not found: $FILE, skipping..."
      continue
    fi
    # Use absolute paths for file arguments
    python3 "${UTILS_DIR}/sing_train.py" \
      --train "$FILE" \
      --test_embeddings "${TEST_EMB}" \
      --test_labels "${TEST_LABELS}" \
      --split_dev
    done
  done
done

