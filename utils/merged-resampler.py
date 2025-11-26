import numpy as np
import os
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, CondensedNearestNeighbour
from imblearn.combine import SMOTEENN
import argparse

# Get the repository root directory (parent of utils/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

def load_embeddings_and_labels(embeddings_file=None, labels_file=None):
    """Load embeddings and labels from NPY files."""
    print("Loading embeddings and labels...")
    
    # Default paths relative to gold_labels/
    if embeddings_file is None:
        embeddings_file = os.path.join(REPO_ROOT, 'gold_labels', 'train_embeddings.npy')
    elif not os.path.isabs(embeddings_file):
        embeddings_file = os.path.join(REPO_ROOT, 'gold_labels', embeddings_file)
    
    if labels_file is None:
        labels_file = os.path.join(REPO_ROOT, 'gold_labels', 'train_labels.npy')
    elif not os.path.isabs(labels_file):
        labels_file = os.path.join(REPO_ROOT, 'gold_labels', labels_file)
    
    embeddings = np.load(embeddings_file)
    labels = np.load(labels_file)
    return embeddings, labels

def create_binary_labels(labels):
    """Convert multilabel to binary (any label present = 1, else 0)."""
    binary = (labels.sum(axis=1) > 0).astype(float)
    return binary

def create_multiclass_labels(labels):
    """Convert multilabel to multiclass string representation."""
    multiclass = [''.join(map(str, row.astype(int))) for row in labels]
    return np.array(multiclass)

def apply_resampling(X, y, resampler, method_name):
    """Apply a resampling technique and return resampled data."""
    print(f"Applying {method_name}...")
    try:
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        dist = Counter(y_resampled)
        print(f"Resampled distribution for {method_name}: {dist}")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"Error applying {method_name}: {e}")
        return None, None

def save_resampled_data(X, y, output_dir, prefix, method_name):
    """Save resampled embeddings and labels to NPY files."""
    os.makedirs(output_dir, exist_ok=True)
    embeddings_file = os.path.join(output_dir, f"{prefix}_{method_name}_embeddings.npy")
    labels_file = os.path.join(output_dir, f"{prefix}_{method_name}_labels.npy")
    np.save(embeddings_file, X)
    np.save(labels_file, y)
    print(f"Saved {embeddings_file} and {labels_file}")

def main():
    parser = argparse.ArgumentParser(description='Apply resampling techniques to embeddings')
    parser.add_argument('--embeddings', type=str, default=None, 
                        help='Input embeddings file (relative to gold_labels/)')
    parser.add_argument('--labels', type=str, default=None,
                        help='Input labels file (relative to gold_labels/)')
    parser.add_argument('--output_dir', type=str, default='resampled_data',
                        help='Output directory for resampled data (relative to data/)')
    args = parser.parse_args()
    
    # Set default output directory relative to data/
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(REPO_ROOT, 'data', args.output_dir)
    
    # Load data
    X, y_multilabel = load_embeddings_and_labels(args.embeddings, args.labels)
    
    # Define resampling techniques
    resampling_methods = {
        'RandomUnderSampler': RandomUnderSampler(random_state=42),
        'SMOTE': SMOTE(random_state=42, k_neighbors=5),
        'ADASYN': ADASYN(random_state=42, n_neighbors=5),
        'RandomOverSampler': RandomOverSampler(random_state=42),
        'SMOTEENN': SMOTEENN(random_state=42),
        'CondensedNearestNeighbour': CondensedNearestNeighbour(random_state=42),
        'TomekLinks': TomekLinks(),
    }
    
    # Process binary labels
    print("\nProcessing binary labels...")
    y_binary = create_binary_labels(y_multilabel)
    
    for method_name, resampler in resampling_methods.items():
        X_resampled, y_resampled = apply_resampling(X, y_binary, resampler, method_name)
        if X_resampled is not None and y_resampled is not None:
            save_resampled_data(X_resampled, y_resampled, args.output_dir, 'binary', method_name)
    
    # Process multiclass labels
    print("\nProcessing multiclass labels...")
    y_multiclass = create_multiclass_labels(y_multilabel)
    
    for method_name, resampler in resampling_methods.items():
        X_resampled, y_resampled = apply_resampling(X, y_multiclass, resampler, method_name)
        if X_resampled is not None and y_resampled is not None:
            save_resampled_data(X_resampled, y_resampled, args.output_dir, 'multiclass', method_name)
    
    print(f"\nResampling completed. Resampled files saved to: {os.path.abspath(args.output_dir)}/")

if __name__ == "__main__":
    main()

