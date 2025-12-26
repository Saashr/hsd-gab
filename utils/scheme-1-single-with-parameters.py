import os
import json
import argparse
import numpy as np
from itertools import product

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# -----------------------------
# Helper: refine labels by threshold
# -----------------------------
def refine_labels(model, X, threshold):
    """
    Convert decision scores into binary predictions using a confidence threshold.
    """
    scores = model.decision_function(X)
    preds = (scores >= threshold).astype(int)

    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    # Handle "NONE" case (no label predicted)
    none_mask = preds.sum(axis=1) == 0
    if none_mask.any():
        none_col = np.zeros((preds.shape[0], 1), dtype=int)
        none_col[none_mask, 0] = 1
        preds = np.hstack([preds, none_col])

    return preds

# -----------------------------
# Helper: load single-label .npy
# -----------------------------
def load_single_label_file(path):
    """
    Load embeddings and single-label annotations from a .npy file
    and convert labels into 3D binary vectors.
    """
    data = np.load(path, allow_pickle=True)

    label_to_vec = {
        "HD":   [1, 0, 0],
        "CV":   [0, 1, 0],
        "VO":   [0, 0, 1],
        "NONE": [0, 0, 0],
        "None": [0, 0, 0],
    }

    embeddings, labels = [], []
    for emb, tag in data:
        embeddings.append(np.array(emb, dtype=float))
        labels.append(label_to_vec.get(str(tag), [0, 0, 0]))

    return np.vstack(embeddings), np.array(labels, dtype=int)

# -----------------------------
# Evaluate predictions
# -----------------------------
def evaluate_predictions(preds, gold, label_names):
    """
    Compute per-label, macro, micro F1 and subset accuracy.
    """
    preds = preds[:, :3]
    gold  = gold[:, :3]

    per_label = {}
    for i, name in enumerate(label_names):
        tp = np.sum((preds[:, i] == 1) & (gold[:, i] == 1))
        fp = np.sum((preds[:, i] == 1) & (gold[:, i] == 0))
        fn = np.sum((preds[:, i] == 0) & (gold[:, i] == 1))

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_label[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return {
        "per_label": per_label,
        "macro_f1": float(f1_score(gold, preds, average="macro")),
        "micro_f1": float(f1_score(gold, preds, average="micro")),
        "accuracy": float(np.mean(np.all(preds == gold, axis=1)))
    }

# -----------------------------
# Train, tune, and evaluate a single file
# -----------------------------
def run_single_file(file_path, test_embeddings, test_labels, thresholds):
    print(f"\n[INFO] Processing file: {file_path}")
    X, y = load_single_label_file(file_path)

    label_names = ["HD", "CV", "VO"]

    # Train/dev split
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=0.22, random_state=42
    )

    # ------------------------------------------------
    # Hyperparameter tuning (macro-F1 on dev set)
    # ------------------------------------------------
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"]
    }

    best_params = None
    best_dev_f1 = -1.0

    for C, class_weight in product(param_grid["C"], param_grid["class_weight"]):
        params = {
            "C": C,
            "tol": 1e-3,
            "loss": "squared_hinge",
            "dual": False,
            "class_weight": class_weight
        }

        try:
            base = LinearSVC(**params, max_iter=10000)
            model = OneVsRestClassifier(base)
            model.fit(X_train, y_train)

            preds_dev = (model.decision_function(X_dev) >= 0).astype(int)
            dev_macro_f1 = f1_score(y_dev, preds_dev[:, :3], average="macro")

            if dev_macro_f1 > best_dev_f1:
                best_dev_f1 = dev_macro_f1
                best_params = params

        except Exception as e:
            print(f"[WARN] Skipping params {params}: {e}")

    print(f"→ Best dev macro-F1: {best_dev_f1:.4f}")
    print(f"→ Best hyperparameters: {best_params}")

    # ------------------------------------------------
    # Threshold tuning (macro-F1 on dev set)
    # ------------------------------------------------
    base_model = LinearSVC(**best_params, max_iter=10000)
    model = OneVsRestClassifier(base_model)
    model.fit(X_train, y_train)

    best_threshold = None
    best_thr_f1 = -1.0

    for t in thresholds:
        preds_dev = refine_labels(model, X_dev, t)[:, :3]
        macro_f1 = f1_score(y_dev, preds_dev, average="macro")

        if macro_f1 > best_thr_f1:
            best_thr_f1 = macro_f1
            best_threshold = t

    print(f"→ Best threshold: {best_threshold} (dev macro-F1={best_thr_f1:.4f})")

    # ------------------------------------------------
    # Retrain on full training data
    # ------------------------------------------------
    X_full = np.vstack([X_train, X_dev])
    y_full = np.vstack([y_train, y_dev])

    final_model = OneVsRestClassifier(
        LinearSVC(**best_params, max_iter=10000)
    )
    final_model.fit(X_full, y_full)

    # ------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------
    preds_test = refine_labels(final_model, test_embeddings, best_threshold)
    metrics = evaluate_predictions(preds_test, test_labels, label_names)

    return {
        "file": file_path,
        "hyperparameters": best_params,
        "best_threshold": best_threshold,
        "results": metrics
    }

# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Run evaluation on a single file")
    args = parser.parse_args()

    # Load test data
    test_embeddings = np.load(os.path.join(REPO_ROOT, "gold_labels", "test_embeddings.npy"))
    test_labels = np.load(os.path.join(REPO_ROOT, "gold_labels", "test_labels.npy"))

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    # ----------------------
    # Single file mode
    # ----------------------
    if args.file:
        result = run_single_file(args.file, test_embeddings, test_labels, thresholds)
        out_path = os.path.join(REPO_ROOT, "results", "single_file_test.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[INFO] Saved result to {out_path}\n")
        return

    # ----------------------
    # Full scheme run
    # ----------------------
    schemes = {
        "1_to_1": "sing_label_data",
        "2_to_1": "sing_label_data2_1",
        "3_to_1": "sing_label_data3_1",
    }

    results_root = os.path.join(REPO_ROOT, "results", "single_label")
    os.makedirs(results_root, exist_ok=True)

    for name, subdir in schemes.items():
        print(f"\n=== RUNNING SCHEME {name} ===")
        data_dir = os.path.join(REPO_ROOT, "data", subdir)

        scheme_results = {}
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith("_single_label.npy"):
                fpath = os.path.join(data_dir, fname)
                scheme_results[fname] = run_single_file(
                    fpath, test_embeddings, test_labels, thresholds
                )

        out_path = os.path.join(results_root, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(scheme_results, f, indent=2)

        print(f"[INFO] Saved: {out_path}")

if __name__ == "__main__":
    main()

