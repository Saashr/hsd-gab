# Training Plan

This document provides a comprehensive guide for setting up and running different training schemes using `main.py`. It explains the various configurations, training modes, and how to execute them.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Training Settings](#training-settings)
4. [Training Modes](#training-modes)
5. [Training Configurations](#training-configurations)
6. [Batch Training](#batch-training)
7. [Understanding Outputs](#understanding-outputs)

## Prerequisites

Before running any training, ensure you have:

1. **Environment setup**: 
   - Conda environment 'HSD' created and activated
   - All dependencies installed from `requirements.txt`
   - See [Environment Setup](#environment-setup) section below

2. **Completed initial setup**: Run `./setup.sh` to extract embeddings and generate resampled datasets

3. **Required directories**:
   - `gold_labels/` - Contains train/test embeddings and labels
   - `data/` - Contains resampled datasets and single-label datasets
   - `logs/` - Will contain training outputs and results

4. **Required files**:
   - `gold_labels/train_embeddings.npy` and `gold_labels/train_labels.npy`
   - `gold_labels/test_embeddings.npy` and `gold_labels/test_labels.npy`

## Environment Setup

### Step 1: Load Conda Module (if on HPC/Cluster)

If you're on a cluster or HPC system, load the conda module first:

```bash
module load conda
```

### Step 2: Create Conda Environment

Create a new conda environment named 'HSD':

```bash
conda create -n HSD python=3.8 -y
conda activate HSD
```

### Step 3: Install Dependencies

Install all required packages from the requirements file:

```bash
pip install -r requirements.txt
```

**Alternative installation method** (if you prefer conda for PyTorch):

```bash
# For CPU-only systems:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# OR for GPU systems (adjust CUDA version as needed):
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Then install the rest via pip:
pip install transformers scikit-learn pandas numpy imbalanced-learn
```

### Step 4: Verify Installation

Verify that all packages are installed correctly:

```bash
python3 -c "import torch; import transformers; import sklearn; import pandas; import numpy; import imblearn; print('All packages imported successfully!')"
```

### Step 5: Activate Environment for Each Session

**Important**: Each time you start a new terminal session, activate the environment:

```bash
module load conda  # If on HPC/cluster
conda activate HSD
```

### Troubleshooting Environment Setup

**Issue: "command not found: conda"**
- Load the conda module: `module load conda`
- Or initialize conda: `source ~/anaconda3/etc/profile.d/conda.sh` (adjust path as needed)

**Issue: "No module named 'transformers'" or similar**
- Ensure you're in the HSD environment: `conda activate HSD`
- Reinstall: `pip install -r requirements.txt`

**Issue: PyTorch CUDA errors**
- Install the correct PyTorch version for your system
- For CPU-only: Use `cpuonly` variant
- For GPU: Match CUDA version with your system

## Initial Setup

**Note**: Ensure your HSD conda environment is activated before proceeding:
```bash
module load conda  # If on HPC/cluster
conda activate HSD
```

### Step 1: Extract Embeddings

If you haven't already, extract embeddings from your TSV files (can be done using scripts in `utils` or by running `bash setup.sh`):

```bash
# Extract training embeddings
python3 utils/get-embeddings.py --input data/ghc_train.tsv

# Extract test embeddings
python3 utils/get-embeddings.py --input data/ghc_test.tsv
```

### Step 2: Generate Resampled Datasets

Apply resampling techniques to create balanced datasets:

```bash
python3 utils/merged-resampler.py \
  --embeddings gold_labels/train_embeddings.npy \
  --labels gold_labels/train_labels.npy \
  --output_dir data/resampled_data
```

### Step 3: (Optional) Generate Single-Label Datasets

For single-label classification experiments:

```bash
./convert_multilabel_to_single.sh
```

## Training Settings

The `--setting` parameter determines how labels are processed:

### 1. Binary Classification (`--setting binary`)

**Purpose**: Classify samples as having any hate speech label (1) or no hate speech (0)

**Data Required**: Files from `data/resampled_data/` with `binary_*` prefix

**Example**:
```bash
python3 utils/main.py \
  --data_dir resampled_data \
  --setting binary \
  --labels 8
```

**What it does**:
- Uses binary resampled datasets (e.g., `binary_SMOTE_embeddings.npy`)
- Converts multilabel problem to binary (any label = 1, none = 0)
- Labels: 2 classes (hate speech / no hate speech)

### 2. Multiclass Classification (`--setting multiclass`)

**Purpose**: Classify samples into one of 8 multiclass categories (combinations of HD, CV, VO)

**Data Required**: Files from `data/resampled_data/` with `multiclass_*` prefix

**Example**:
```bash
python3 utils/main.py \
  --data_dir resampled_data \
  --setting multiclass \
  --labels 8
```

**What it does**:
- Uses multiclass resampled datasets (e.g., `multiclass_SMOTE_embeddings.npy`)
- Treats each combination of HD/CV/VO as a separate class (e.g., "100", "010", "111")
- Labels: 8 classes (all combinations of 3 binary labels)

### 3. Baseline (`--setting baseline`)

**Purpose**: Train on original (non-resampled) training data

**Data Required**: Files from `gold_labels/`

**Example**:
```bash
python3 utils/main.py \
  --data_dir baseline \
  --setting baseline \
  --baseline_data_dir gold_labels \
  --labels 4
```

**What it does**:
- Uses original `gold_labels/train_embeddings.npy` and `train_labels.npy`
- No resampling applied
- Baseline comparison for resampling techniques

### 4. Single-Label Classification (`--setting single`)

**Purpose**: Classify into a single label category (HD, CV, VO, or None)

**Data Required**: Files from `data/sing_label_data/` with `*_single_label.npy` suffix

**Example**:
```bash
python3 utils/main.py \
  --data_dir sing_label_data \
  --setting single \
  --file data/sing_label_data/multiclass_SMOTE_single_label.npy \
  --confidence
```

**What it does**:
- Uses single-label converted datasets
- Each sample belongs to exactly one of: HD, CV, VO, or None
- Labels: 4 classes

## Training Modes

Training modes determine how the model is trained and evaluated:

### Mode 1: Default Training (No Dev Set)

**Use Case**: When you don't need to tune hyperparameters or thresholds

**Command**:
```bash
python3 utils/main.py \
  --data_dir resampled_data \
  --setting binary \
  --labels 8
```

**Behavior**:
- Uses all training data for training
- No dev set split
- Uses fixed hyperparameters: `C=10, tol=0.001, loss='squared_hinge', penalty='l2', dual=False`
- Evaluates on test set with multiple confidence thresholds (0.3-0.9)
- Reports best threshold based on macro F1

### Mode 2: Dev Set with Confidence Threshold Tuning (`--split_dev --use_confidence_dev`)

**Use Case**: Find optimal confidence threshold using dev set

**Command**:
```bash
python3 utils/main.py \
  --data_dir sing_label_data \
  --setting single \
  --split_dev \
  --use_confidence_dev \
  --confidence \
  --file data/sing_label_data/multiclass_SMOTE_single_label.npy
```

**Behavior**:
- Splits training data: 77.8% train, 22.2% dev (test_size=0.222)
- Trains on train split with fixed hyperparameters
- Tests confidence thresholds from 0.3 to 0.9 (step 0.05) on dev set
- Selects threshold with best macro F1 on dev set
- Retrains on combined train+dev data
- Evaluates on test set using best threshold
- Saves confidence scores to `logs/confidence_scores/`

**Best for**: Single-label classification where threshold tuning is critical

### Mode 3: Dev Set with Hyperparameter Search (`--split_dev --hyperparam_search`)

**Use Case**: Find optimal SVM hyperparameters (see paper for latest)

**Command**:
```bash
python3 utils/main.py \
  --data_dir resampled_data \
  --setting multiclass \
  --split_dev \
  --hyperparam_search \
  --labels 8
```

**Behavior**:
- Splits training data into train/dev (77.8%/22.2%)
- Performs grid search over hyperparameters:
  - `C`: [0.001, 0.01, 0.1, 1, 10, 100]
  - `tol`: [1e-4, 1e-3, 1e-2]
  - `loss`: ['hinge', 'squared_hinge']
  - `dual`: [True, False]
- Selects best hyperparameters based on dev set macro F1
- Retrains on combined train+dev data with best hyperparameters
- Evaluates on test set

**Best for**: Finding optimal model hyperparameters for our specific dataset

### Mode 4: Custom Model Parameters (`--model_params`)

**Use Case**: Use specific hyperparameters instead of defaults

**Command**:
```bash
python3 utils/main.py \
  --data_dir resampled_data \
  --setting binary \
  --model_params "{'C': 1.0, 'tol': 0.0001, 'loss': 'hinge', 'penalty': 'l2', 'dual': True}" \
  --labels 8
```

**Behavior**:
- Uses our specified hyperparameters
- No dev set split
- Trains on all training data
- Evaluates on test set

**Best for**: Using known good hyperparameters or testing specific configurations

### Mode 5: Confidence Score Output (`--confidence`)

**Use Case**: Save decision function scores for analysis

**Command**:
```bash
python3 utils/main.py \
  --data_dir resampled_data \
  --setting binary \
  --confidence \
  --labels 8
```

**Behavior**:
- Saves confidence scores (decision function values) for each test sample
- Files saved to `logs/confidence_scores/confidence_<Label>.txt`
- Each file contains scores for one label classifier

**Best for**: Analyzing model confidence, error analysis, or threshold selection

## Training Configurations

### Configuration 1: Binary Classification with Resampled Data

```bash
python3 utils/main.py \
  --data_dir resampled_data \
  --setting binary \
  --labels 8
```

**Input**: `data/resampled_data/binary_*_embeddings.npy` and `binary_*_labels.npy`  
**Output**: Results printed to console, best threshold selected

### Configuration 2: Multiclass Classification with Hyperparameter Tuning

```bash
python3 utils/main.py \
  --data_dir resampled_data \
  --setting multiclass \
  --split_dev \
  --hyperparam_search \
  --labels 8
```

**Input**: `data/resampled_data/multiclass_*_embeddings.npy` and `multiclass_*_labels.npy`  
**Output**: Grid search results, best hyperparameters, final evaluation

### Configuration 3: Single-Label with Confidence Threshold Tuning

```bash
python3 utils/main.py \
  --data_dir sing_label_data \
  --setting single \
  --split_dev \
  --use_confidence_dev \
  --confidence \
  --file data/sing_label_data/multiclass_SMOTE_single_label.npy
```

**Input**: Single-label dataset file  
**Output**: Best confidence threshold, confidence scores saved, final evaluation

### Configuration 4: Baseline Comparison

```bash
python3 utils/main.py \
  --data_dir baseline \
  --setting baseline \
  --baseline_data_dir gold_labels \
  --labels 4 \
  --confidence
```

**Input**: `gold_labels/train_embeddings.npy` and `train_labels.npy`  
**Output**: Baseline performance metrics

### Configuration 5: Single File Processing

```bash
python3 utils/main.py \
  --data_dir sing_label_data \
  --setting single \
  --file data/sing_label_data/multiclass_ADASYN_single_label.npy \
  --split_dev \
  --use_confidence_dev \
  --confidence
```

**Input**: Specific NPY file  
**Output**: Results for that specific dataset

## Batch Training

### Process All Resampled Datasets

Use `--data_dir all` to process multiple directories:

```bash
python3 utils/main.py \
  --data_dir all \
  --setting binary \
  --labels 8
```

This processes:
- `resampled_data`
- `resampled_data2_1`
- `resampled_data3_1`
- `sing_label_data`
- `sing_label_data2_1`
- `sing_label_data3_1`

### Using Shell Scripts for Batch Processing

**Multi-label evaluation**:
```bash
./multi_label_eval.sh
```

**Single-label evaluation with confidence tuning**:
```bash
./run_all_single_label.sh
```

**General batch evaluation**:
```bash
./run_all_evaluations.sh
```

## Understanding Outputs

### Console Output

The script prints:
- Training/dev/test sample counts
- Best threshold (if using `--use_confidence_dev`)
- Best hyperparameters (if using `--hyperparam_search`)
- Per-label metrics: Precision, Recall, F1 for each label
- Average metrics: Precision, Recall, F1 across labels
- Micro F1: Overall F1 score across all samples
- Macro F1: Average F1 score across labels

### Saved Files

**Confidence Scores** (`logs/confidence_scores/`):
- `confidence_HD.txt`, `confidence_CV.txt`, `confidence_VO.txt`, etc.
- Contains decision function scores for each test sample

**Results** (if redirected):
- Save output to file: `python3 utils/main.py ... > logs/results/my_experiment.txt`

### Evaluation Metrics

The script evaluates at multiple confidence thresholds: [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

For each threshold, it calculates:
- Per-label Precision, Recall, F1
- Average Precision, Recall, F1
- Micro F1 (sample-level)
- Macro F1 (label-level)

The **best threshold** is selected based on highest macro F1 score.

## Complete Training Workflow Examples

### Example 1: Quick Binary Classification

```bash
# Setup (one time)
./setup.sh

# Train binary classifier
python3 utils/main.py \
  --data_dir resampled_data \
  --setting binary \
  --labels 8
```

### Example 2: Comprehensive Single-Label Analysis

```bash
# Setup
./setup.sh

# Generate single-label datasets
./convert_multilabel_to_single.sh

# Train with confidence tuning
python3 utils/main.py \
  --data_dir sing_label_data \
  --setting single \
  --split_dev \
  --use_confidence_dev \
  --confidence \
  --file data/sing_label_data/multiclass_SMOTE_single_label.npy
```

### Example 3: Full Hyperparameter Search

```bash
# Setup
./setup.sh

# Run hyperparameter search
python3 utils/main.py \
  --data_dir resampled_data \
  --setting multiclass \
  --split_dev \
  --hyperparam_search \
  --labels 8 \
  > logs/hyperparam_search_results.txt
```

### Example 4: Compare All Resampling Methods

```bash
# Setup
./setup.sh

# Run all binary experiments
python3 utils/main.py \
  --data_dir all \
  --setting binary \
  --labels 8 \
  > logs/all_binary_results.txt
```

## Tips and Best Practices

1. **Start with baseline**: Always run baseline first to establish a comparison point
   ```bash
   python3 utils/main.py --data_dir baseline --setting baseline --baseline_data_dir gold_labels
   ```

2. **Use dev set for tuning**: When tuning hyperparameters or thresholds, always use `--split_dev`

3. **Save outputs**: Redirect output to files for later analysis
   ```bash
   python3 utils/main.py ... > logs/my_experiment.txt 2>&1
   ```

4. **Batch processing**: Use shell scripts for running multiple experiments systematically

5. **Confidence scores**: Enable `--confidence` when you need to analyze model predictions in detail

6. **Memory considerations**: Hyperparameter search can be memory-intensive; monitor system resources

7. **Reproducibility**: The random_state is fixed (42), ensuring reproducible results

## Troubleshooting

**Error: "Required test files not found"**
- Ensure `gold_labels/test_embeddings.npy` and `test_labels.npy` exist
- Run `./setup.sh` if they're missing

**Error: "Data directory does not exist"**
- Check that `data/resampled_data/` exists
- Run `./setup.sh` or manually run `merged-resampler.py`

**Error: "No *_single_label.npy files found"**
- Generate single-label datasets: `./convert_multilabel_to_single.sh`

**Error: "No module named 'transformers'" or import errors**
- Ensure HSD environment is activated: `conda activate HSD`
- Reinstall dependencies: `pip install -r requirements.txt`

**Low performance**
- Try hyperparameter search: `--split_dev --hyperparam_search`
- Experiment with different resampling methods
- Check data quality and label distribution

## Additional Resources
- See `README.md` for general repository information
- Check `utils/main.py` for implementation details and for argument parsin aids.
- Review shell scripts for batch processing examples

