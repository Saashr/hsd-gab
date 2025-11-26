# hsd-gab

## Overview

This repository contains code, data, and results for an NLP project investigating the impact of various resampling techniques on hate speech detection in multiclass and multilabel datasets. The research focuses on mitigating class imbalance and label sparsity to improve the prediction of minority classes through data preprocessing methods such as resampling and synthetic example generation. The scripts also facilitate evaluation utilizing SVM binary and multiclass classification.

## Directory Structure

The repository is organized into the following directories:

```
hsd-gab/
├── utils/                    # Utility scripts
│   ├── get-embeddings.py     # Extract BERT embeddings from TSV files
│   ├── merged-resampler.py   # Apply resampling techniques
│   └── main.py               # SVM training and evaluation
├── data/                     # Data files and directories
│   ├── ghc_train.tsv         # Training dataset (Gab Hate Speech Corpus)
│   ├── ghc_test.tsv          # Test dataset (Gab Hate Speech Corpus)
│   ├── resampled_data/       # Resampled datasets (binary and multiclass)
│   ├── resampled_data2_1/    # Additional resampled datasets
│   ├── resampled_data3_1/    # Additional resampled datasets
│   ├── sing_label_data/      # Single-label converted datasets
│   ├── sing_label_data2_1/   # Additional single-label datasets
│   └── sing_label_data3_1/   # Additional single-label datasets
├── gold_labels/              # Gold standard labels and embeddings
│   ├── train_embeddings.npy  # Training embeddings
│   ├── train_labels.npy      # Training labels
│   ├── test_embeddings.npy   # Test embeddings
│   └── test_labels.npy       # Test labels
└── logs/                     # Training logs and outputs
    ├── results-hsd-gab/      # Classification results
    ├── confidence_scores/    # Confidence score outputs
    └── *.txt                 # Various log files
```

## Key Files

### Utility Scripts (`utils/`)

- **`get-embeddings.py`**  
  Extracts BERT-based-uncased embeddings from TSV files. Reads from `data/` directory and outputs embeddings and labels to `gold_labels/` directory. Supports both training and test datasets.

- **`merged-resampler.py`**  
  Applies various resampling techniques (RandomUnderSampler, SMOTE, ADASYN, RandomOverSampler, SMOTEENN, CondensedNearestNeighbour, TomekLinks) to address class imbalance in both binary and multiclass settings. Reads from `gold_labels/` and outputs resampled datasets to `data/resampled_data/` as NPY files.

- **`main.py`**  
  Main training and evaluation script. Reads NPY arrays from `data/` and `gold_labels/` directories, trains SVM classifiers, and outputs results. Supports binary, multiclass, and single-label classification modes. See argument parser details for adjusting parameters. Usually run in batches using `run_all_evaluations.sh` or `run_all_single_label.sh`.

### Data Files (`data/`)

- **`ghc_train.tsv`** and **`ghc_test.tsv`**  
  Training and testing datasets from the Gab Hate Speech Corpus: https://osf.io/edua3/

- **`resampled_data/`, `resampled_data2_1/`, `resampled_data3_1/`**  
  Directories containing resampled training datasets (both binary and multiclass formats).

- **`sing_label_data/`, `sing_label_data2_1/`, `sing_label_data3_1/`**  
  Directories containing single-label converted datasets for single-label classification experiments.

### Gold Labels (`gold_labels/`)

- **`train_embeddings.npy`** and **`train_labels.npy`**  
  Training embeddings and labels extracted from the training dataset.

- **`test_embeddings.npy`** and **`test_labels.npy`**  
  Test embeddings and labels extracted from the test dataset.

### Logs (`logs/`)

- **`results-hsd-gab/`**  
  Directory containing evaluation results and metrics.

- **`confidence_scores/`**  
  Directory containing confidence score outputs for each label classifier.

## Setup

### Environment Setup

1. **Load conda module** (if on a cluster/HPC system):
   ```bash
   module load conda
   ```

2. **Create conda environment**:
   ```bash
   conda create -n HSD python=3.8 -y
   conda activate HSD
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, if using conda for some packages:
   ```bash
   conda install pytorch torchvision torchaudio cpuonly -c pytorch  # For CPU-only
   # OR
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch  # For GPU
   pip install transformers scikit-learn pandas numpy imbalanced-learn
   ```

### Data Setup

1. **Initial Setup**: Run the setup script to extract embeddings and apply resampling:
   ```bash
   ./setup.sh
   ```
   This will:
   - Extract embeddings from `data/ghc_train.tsv` and `data/ghc_test.tsv`
   - Save embeddings and labels to `gold_labels/`
   - Apply resampling techniques and save results to `data/resampled_data/`

## Usage

### Preprocessing

1. **Extract Embeddings**: Run `get-embeddings.py` to extract embeddings from raw TSV files:
   ```bash
   python3 utils/get-embeddings.py --input data/ghc_train.tsv
   python3 utils/get-embeddings.py --input data/ghc_test.tsv
   ```
   Embeddings will be saved to `gold_labels/` directory.

2. **Apply Resampling**: Use `merged-resampler.py` to apply resampling techniques:
   ```bash
   python3 utils/merged-resampler.py \
     --embeddings gold_labels/train_embeddings.npy \
     --labels gold_labels/train_labels.npy \
     --output_dir data/resampled_data
   ```

### Training and Evaluation

1. **Single Evaluation**: Use `main.py` to train and test SVM classifier:
   ```bash
   python3 utils/main.py \
     --data_dir resampled_data \
     --setting binary \
     --labels 8
   ```

2. **Batch Evaluation**: Run batch evaluations using shell scripts:
   ```bash
   # Multi-label evaluation
   ./multi_label_eval.sh
   
   # Single-label evaluation with confidence thresholds
   ./run_all_single_label.sh
   
   # General evaluations
   ./run_all_evaluations.sh
   ```

3. **Results**: Evaluate performance and analyze results in `logs/results-hsd-gab/`

### Thresholding Label Training

1. Use `main.py` to train and test SVM classifier. See argument parsing logic to adjust the settings:
   ```bash
   python3 utils/main.py \
     --data_dir sing_label_data \
     --setting single \
     --use_confidence_dev \
     --confidence \
     --file data/sing_label_data/multiclass_SMOTE_single_label.npy
   ```

2. Run using a variety of confidence thresholds using `run_all_evaluations.sh` or `run_all_single_label.sh`

3. Alternatively, use the dev set to set the threshold for the test set by using the `--split_dev` flag

## Command Line Arguments

### `get-embeddings.py`
- `--input`: Input TSV file path (relative to `data/` or absolute)
- `--output_embeddings`: Output embeddings file path (relative to `gold_labels/` or absolute)
- `--output_labels`: Output labels file path (relative to `gold_labels/` or absolute)

### `merged-resampler.py`
- `--embeddings`: Input embeddings file (relative to `gold_labels/` or absolute)
- `--labels`: Input labels file (relative to `gold_labels/` or absolute)
- `--output_dir`: Output directory for resampled data (relative to `data/` or absolute)

### `main.py`
- `--data_dir`: Data directory (relative to `data/` or absolute)
- `--setting`: Classification setting (`binary`, `multiclass`, `baseline`, or `single`)
- `--confidence`: Save confidence scores
- `--threshold`: Confidence threshold (default: 0.7)
- `--labels`: Number of labels (default: 4)
- `--baseline_data_dir`: Baseline data directory (for baseline setting)
- `--split_dev`: Split data into train/dev sets
- `--hyperparam_search`: Perform hyperparameter grid search
- `--use_confidence_dev`: Use dev set to select best confidence threshold
- `--file`: Path to a specific .npy file to load directly

## Notes

- All paths in scripts are relative to the repository root and work cross-platform
- The repository root is automatically detected by all scripts
- Output directories are created automatically if they don't exist
- All scripts support both relative paths (from repo root) and absolute paths
