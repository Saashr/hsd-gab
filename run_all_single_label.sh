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

# Directories to process
DIRS=("sing_label_data" "sing_label_data2_1" "sing_label_data3_1")

# Mode flags (set exactly one to true)
HYPERPARAM_SEARCH=false
CONF_DEV=true

for DIR in "${DIRS[@]}"; do
  DIR_PATH="${DATA_DIR}/${DIR}"
  echo "== Processing directory: $DIR_PATH =="

  if [[ ! -d "$DIR_PATH" ]]; then
    echo "WARNING: Directory $DIR_PATH not found, skipping."
    continue
  fi

  # Find all *_single_label.npy files
  shopt -s nullglob
  FILES=("${DIR_PATH}"/*_single_label.npy)
  shopt -u nullglob

  if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No *_single_label.npy files found in $DIR_PATH"
    continue
  fi

  # Process each file individually
  for FILE in "${FILES[@]}"; do
    FILE_NAME=$(basename "$FILE")
    PREFIX="${FILE_NAME%_single_label.npy}"
    # Use absolute path for file argument
    echo "== Processing file: $FILE_NAME =="

    # --- Step 1: Dev set / confidence mode ---
    if [[ "$CONF_DEV" = true ]]; then
      echo "Running main.py (confidence dev set) on $FILE_NAME"
      DEV_LOG=$(mktemp)
      python3 "${UTILS_DIR}/main.py" \
        --data_dir "$DIR" \
        --setting single \
        --use_confidence_dev \
        --confidence \
        --file "$FILE" > "$DEV_LOG" 2>&1

      BEST_THRESH=$(grep "\[INFO\] Best confidence threshold on dev set:" "$DEV_LOG" | awk '{print $7}')
      if [[ -z "$BEST_THRESH" ]]; then
        echo "ERROR: Could not extract best confidence threshold for $FILE_NAME"
        cat "$DEV_LOG"
        rm "$DEV_LOG"
        continue
      fi
      echo "[INFO] Best confidence threshold for $FILE_NAME = $BEST_THRESH"
      rm "$DEV_LOG"
    fi

    # --- Step 2: Retrain / final evaluation ---
    echo "Running main.py (final single-label) on $FILE_NAME with threshold=$BEST_THRESH"
    python3 "${UTILS_DIR}/main.py" \
      --data_dir "$DIR" \
      --setting single \
      --confidence \
      --threshold "$BEST_THRESH" \
      --file "$FILE" \
      $( [[ "$HYPERPARAM_SEARCH" = true ]] && echo "--hyperparam_search" )
  done
done

