# utils/resample_multiclass_11.py
import numpy as np
import os
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour, TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN

# Figure out repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Paths for input and output files
INPUT_EMBEDDINGS = os.path.join(REPO_ROOT, "gold_labels", "train_embeddings.npy")
INPUT_LABELS = os.path.join(REPO_ROOT, "gold_labels", "train_labels.npy")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "resampled_data")
TXT_FILE = os.path.join(REPO_ROOT, "data", "resampled_data_multiclass11.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(TXT_FILE, 'w') as f:
    # Load embeddings and labels
    print("Loading embeddings and labels...")
    f.write("Loading embeddings and labels...\n")
    X_train = np.load(INPUT_EMBEDDINGS)
    y_train = np.load(INPUT_LABELS)

    def convert_to_multiclass(y):
        # rows like [1,0,1] -> "101"
        return np.array(["{}{}{}".format(*row) for row in y])

    y_multiclass = convert_to_multiclass(y_train)

    # 1:1 / fully balanced resampling strategies
    resampling_techniques = {
        "RandomUnderSampler": RandomUnderSampler(random_state=42),
        "CondensedNearestNeighbour": CondensedNearestNeighbour(random_state=42),
        "TomekLinks": TomekLinks(),
        "SMOTE": SMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "RandomOverSampler": RandomOverSampler(random_state=42),
        "SMOTEENN": SMOTEENN(random_state=42),
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
