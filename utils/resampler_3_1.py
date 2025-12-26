# utils/resample_multiclass_31.py
import os
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN

# Figure out repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Paths for input and output files
INPUT_EMBEDDINGS = os.path.join(REPO_ROOT, "gold_labels", "train_embeddings.npy")
INPUT_LABELS = os.path.join(REPO_ROOT, "gold_labels", "train_labels.npy")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "resampled_data_multiclass31")
TXT_FILE = os.path.join(REPO_ROOT, "data", "resampled_data_multiclass31.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(TXT_FILE, 'w') as f:
    # Load embeddings and labels
    print("Loading embeddings and labels...")
    f.write("Loading embeddings and labels...\n")
    X_train = np.load(INPUT_EMBEDDINGS)
    y_train = np.load(INPUT_LABELS)

    def convert_to_multiclass(y):
        return np.array(["{}{}{}".format(*row) for row in y])

    print(len(y_train))
    f.write(f"Number of training instances: {len(y_train)}\n")
    y_multiclass = convert_to_multiclass(y_train)

    # Your exact 3:1 dicts
    undersampling_dict = {
        '000': 8010, '001': 729, '010': 56, '011': 28,
        '100': 1211, '101': 599, '110': 24, '111': 23
    }
    ratio_3_1 = int(6455 / 2670)
    oversampling_dict = {
        '000': 19366,
        '001': 729 * ratio_3_1,
        '010': 56 * ratio_3_1,
        '011': 28 * ratio_3_1,
        '100': 1211 * ratio_3_1,
        '101': 599 * ratio_3_1,
        '110': 24 * ratio_3_1,
        '111': 23 * ratio_3_1,
    }

    resampling_techniques = {
        "RandomUnderSampler": RandomUnderSampler(random_state=42, sampling_strategy=undersampling_dict),
        "SMOTE": SMOTE(random_state=42, sampling_strategy=oversampling_dict),
        "ADASYN": ADASYN(random_state=42, sampling_strategy=oversampling_dict),
        "RandomOverSampler": RandomOverSampler(random_state=42, sampling_strategy=oversampling_dict),
        "SMOTEENN": SMOTEENN(random_state=42, sampling_strategy=oversampling_dict),
    }

    def resample_data(X, y, technique_name, technique):
        print(f"\nApplying {technique_name}...")
        f.write(f"\nApplying {technique_name}...\n")
        try:
            X_resampled, y_resampled = technique.fit_resample(X, y)
            dist = Counter(y_resampled)
            print(f"Resampled distribution for {technique_name}: {dist}")
            f.write(f"Resampled distribution for {technique_name}: {dist}\n")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Error applying {technique_name}: {e}")
            f.write(f"Error applying {technique_name}: {e}\n")
            return X, y

    print("\nProcessing multiclass labels...")
    f.write("\nProcessing multiclass labels...\n")
    for name, technique in resampling_techniques.items():
        X_resampled, y_resampled = resample_data(X_train, y_multiclass, name, technique)
        np.save(os.path.join(OUTPUT_DIR, f"multiclass_{name}_embeddings.npy"), X_resampled)
        np.save(os.path.join(OUTPUT_DIR, f"multiclass_{name}_labels.npy"), y_resampled)

    print("\nResampling completed. Resampled files saved to:", OUTPUT_DIR)
    f.write("\nResampling completed. Resampled files saved to: " + OUTPUT_DIR + "\n")

