# Hsd-gab

Hsd-gab is a hate speech detection pipeline that converts text into embeddings, mitigates class imbalance through resampling, and trains SVM-based classifiers for multi-label hate speech detection.

---

## Pipeline Overview

The pipeline consists of the following stages:

1. Text to embedding conversion  
2. Dataset resampling (1:1, 2:1, 3:1)  
3. Multi-label to single-label data transformation  
4. Single-label model training and multi-label inference using SVMs  

All experiments use fixed embeddings and LinearSVC classifiers.

---

## Step 1: Embedding Extraction

Raw text data is converted into dense vector embeddings that serve as input features for all downstream tasks.

### Inputs
data/ghc_train.tsv
data/ghc_test.tsv


### Outputs
gold_labels/
├── train_embeddings.npy
├── train_labels.npy
├── test_embeddings.npy
└── test_labels.npy



---

## Step 2: Dataset Resampling

To address class imbalance, the embedding–label pairs are resampled using multiple class-ratio strategies:

- 1:1  
- 2:1  
- 3:1  

### Outputs

data/
├── resampled_data/
├── resampled_data_multiclass11/
├── resampled_data_multiclass21/
└── resampled_data_multiclass31/




Each directory contains summary files describing post-resampling label distributions.

---

## Step 3: Thresholding-Based Multi-Label Scheme (Implemented)

### Overview

The implemented approach converts the multi-label problem into a single-label training task and recovers multi-label predictions at inference time using thresholding.

### Label Space

{HD, CV, VO, NONE}


### Training Data Construction

- Instances with multiple active labels are duplicated, creating one training instance per label.
- Instances with no active labels are assigned the label NONE.
- Duplication is applied only during data preparation.

### Outputs
data/
├── sing_label_data/
├── sing_label_data2_1/
└── sing_label_data3_1/


---

## Step 4: Model Training and Evaluation

A One-vs-Rest LinearSVC classifier is trained on the resampled single-label datasets.

- No thresholding is applied during training.
- At inference time, decision scores are produced for HD, CV, and VO.
- Fixed confidence thresholds are applied independently to each label.
- An instance is labeled NONE when no score exceeds the threshold.

Thresholds are tuned on a development set to optimize macro-averaged F1 score.

### Outputs
results/single_label/.json
logs/confidence_scores/


Evaluation uses standard multilabel metrics, including per-label precision, recall, and macro/micro-averaged F1.

---

## Planned Extension

A multi-label complex classification scheme that treats each label combination as a 
distinct class will be added in a future update.

---

## Notes

- All jobs are designed to run on HPC systems using SLURM.

- The pipeline is modular, allowing individual stages to be rerun independently.

