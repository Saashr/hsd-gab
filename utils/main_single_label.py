import os
import json
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
def refine_labels(model, X, confidence_threshold):
    """
    Take decision_function scores and convert to 0/1 labels
    with a minimum of one label per sample (adds a 'None' column internally).
    """
    decision_function = model.decision_function(X)
    preds = (decision_function >= confidence_threshold).astype(int)

    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    # if a sample has no labels above threshold, mark a "None" column
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
    Expected structure of .npy:
    np.array of [embedding, tag_string] rows,
    where tag_string is one of 'HD', 'CV', 'VO', 'NONE'/'None'.
    """
    data = np.load(path, allow_pickle=True)

    embeddings = []
    labels_vecs = []

    label_to_vec = {
        "HD":   [1, 0, 0],
        "CV":   [0, 1, 0],
        "VO":   [0, 0, 1],
        "NONE": [0, 0, 0],
        "None": [0, 0, 0],
    }

    for row in data:
        emb, tag = row
        embeddings.append(np.array(emb, dtype=float))
        vec = label_to_vec.get(str(tag), [0, 0, 0])
        labels_vecs.append(vec)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels_vecs, dtype=int)

    return embeddings, labels

# -----------------------------
# Helper: evaluate predictions
# -----------------------------
def evaluate_predictions(preds, gold, label_names):
    """
    preds, gold: shape (n_samples, n_labels)
    label_names: list of label strings (e.g. ["HD", "CV", "VO"])
    """
    # align label dimensions
    min_labels = min(preds.shape[1], gold.shape[1])
    preds = preds[:, :min_labels]
    gold = gold[:, :min_labels]

    per_label_metrics = {}
    for i, label in enumerate(label_names[:min_labels]):
        tp = np.sum((preds[:, i] == 1) & (gold[:, i] == 1))
        fp = np.sum((preds[:, i] == 1) & (gold[:, i] == 0))
        fn = np.sum((preds[:, i] == 0) & (gold[:, i] == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_label_metrics[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    avg_precision = float(np.mean([m["precision"] for m in per_label_metrics.values()]))
    avg_recall    = float(np.mean([m["recall"]    for m in per_label_metrics.values()]))
    avg_f1        = float(np.mean([m["f1"]        for m in per_label_metrics.values()]))

    micro_f1 = float(f1_score(gold, preds, average="micro", zero_division=0))
    macro_f1 = float(f1_score(gold, preds, average="macro", zero_division=0))

    # subset accuracy (exact match of all labels)
    accuracy = float(np.mean(np.all(preds == gold, axis=1)))

    return {
        "per_label": per_label_metrics,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }

# -----------------------------
# Single file pipeline
# -----------------------------
def run_single_file(file_path, test_embeddings, test_labels, param_grid, thresholds, random_state=42):
    """
    Runs full pipeline (train/dev split, hyperparam search, threshold tuning, test eval)
    for ONE single-label training file.
    Returns a result dict with hyperparams, threshold, and metrics.
    """
    print(f"\n[INFO] Loading single-label file: {file_path}")
    train_embeddings, train_labels = load_single_label_file(file_path)

    # we work with 3 explicit labels: HD, CV, VO
    label_names = ["HD", "CV", "VO"]

    # train/dev split
    X_train, X_dev, y_train, y_dev = train_test_split(
        train_embeddings,
        train_labels,
        test_size=0.222,
        random_state=random_state,
        stratify=None  # could stratify on something if needed
    )

    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Dev samples: {len(X_dev)}")
    print(f"[INFO] Test samples: {len(test_embeddings)}")

    # ------------------------
    # 1) Hyperparameter search on dev (threshold = 0 baseline)
    # ------------------------
    best_params = None
    best_dev_macro_f1 = -1.0

    print("[INFO] Starting hyperparameter grid search on dev set...")

    for C, tol, loss, dual in product(
        param_grid["C"], param_grid["tol"], param_grid["loss"], param_grid["dual"]
    ):
        params = {
            "C": C,
            "tol": tol,
            "loss": loss,
            "penalty": "l2",
            "dual": dual,
        }

        try:
            base_model = LinearSVC(**params, max_iter=10000, random_state=random_state)
            ovr_model = OneVsRestClassifier(base_model)

            ovr_model.fit(X_train, y_train)
            # default threshold 0 equivalent: decision_function >= 0
            scores_dev = ovr_model.decision_function(X_dev)
            preds_dev = (scores_dev >= 0).astype(int)

            # compute macro F1 on dev
            dev_macro_f1 = f1_score(y_dev, preds_dev, average="macro", zero_division=0)

            if dev_macro_f1 > best_dev_macro_f1:
                best_dev_macro_f1 = dev_macro_f1
                best_params = params

        except Exception as e:
            print(f"[WARN] Skipping params {params} due to error: {e}")
            continue

    print(f"[INFO] Best dev macro F1 from grid search: {best_dev_macro_f1:.4f}")
    print(f"[INFO] Best hyperparameters: {best_params}")

    # ------------------------
    # 2) Threshold tuning on dev with best hyperparameters
    # ------------------------
    print("[INFO] Tuning confidence threshold on dev set...")

    base_model = LinearSVC(**best_params, max_iter=10000, random_state=random_state)
    ovr_model = OneVsRestClassifier(base_model)
    ovr_model.fit(X_train, y_train)

    best_threshold = None
    best_thr_macro_f1 = -1.0

    for thr in thresholds:
        preds_dev_thr_full = refine_labels(ovr_model, X_dev, thr)
        # clip to 3 labels (ignore 'None' column for scoring)
        min_labels = min(preds_dev_thr_full.shape[1], y_dev.shape[1])
        preds_dev_thr = preds_dev_thr_full[:, :min_labels]

        macro_f1_dev_thr = f1_score(y_dev, preds_dev_thr, average="macro", zero_division=0)

        if macro_f1_dev_thr > best_thr_macro_f1:
            best_thr_macro_f1 = macro_f1_dev_thr
            best_threshold = thr

    print(f"[INFO] Best threshold on dev: {best_threshold:.2f} (macro F1={best_thr_macro_f1:.4f})")

    # ------------------------
    # 3) Train final model on train+dev with best hyperparams
    # ------------------------
    X_combined = np.vstack([X_train, X_dev])
    y_combined = np.vstack([y_train, y_dev])

    final_base_model = LinearSVC(**best_params, max_iter=10000, random_state=random_state)
    final_model = OneVsRestClassifier(final_base_model)
    final_model.fit(X_combined, y_combined)

    # ------------------------
    # 4) Evaluate on test set with best threshold
    # ------------------------
    preds_test_full = refine_labels(final_model, test_embeddings, best_threshold)
    metrics = evaluate_predictions(preds_test_full, test_labels, label_names)

    result = {
        "file": os.path.relpath(file_path, REPO_ROOT),
        "best_hyperparameters": best_params,
        "best_threshold": float(best_threshold),
    }
    result.update(metrics)

    return result

# -----------------------------
# Scheme runner (1:1, 2:1, 3:1)
# -----------------------------
def run_scheme(scheme_name, data_subdir, test_embeddings, test_labels, param_grid, thresholds):
    """
    scheme_name: "1_to_1", "2_to_1", "3_to_1"
    data_subdir: e.g. "sing_label_data" (relative to data/)
    """
    data_dir = os.path.join(REPO_ROOT, "data", data_subdir)

    if not os.path.isdir(data_dir):
        print(f"[WARN] Data directory does not exist for {scheme_name}: {data_dir}")
        return {}

    results = {}

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npy"):
            continue
        if not fname.endswith("_single_label.npy"):
            # Only process single-label npy files
            continue

        fpath = os.path.join(data_dir, fname)
        method_name = os.path.splitext(fname)[0]  # e.g. "multiclass_SMOTE_single_label"

        print(f"\n[SCHEME: {scheme_name}] Processing method: {method_name}")
        res = run_single_file(fpath, test_embeddings, test_labels, param_grid, thresholds)
        results[method_name] = res

    return results

# -----------------------------
# Main: run all schemes & save JSON
# -----------------------------
def main():
    # Load test data (gold_labels)
    test_emb_file = os.path.join(REPO_ROOT, "gold_labels", "test_embeddings.npy")
    test_lbl_file = os.path.join(REPO_ROOT, "gold_labels", "test_labels.npy")

    if not (os.path.isfile(test_emb_file) and os.path.isfile(test_lbl_file)):
        raise FileNotFoundError("test_embeddings.npy or test_labels.npy not found in gold_labels/")

    test_embeddings = np.load(test_emb_file)
    test_labels = np.load(test_lbl_file)

    # Hyperparameter grid (as requested)
    param_grid = {
        "C":   [0.001, 0.01, 0.1, 1, 10, 100],
        "tol": [1e-4, 1e-3, 1e-2],
        "loss": ["hinge", "squared_hinge"],
        "dual": [True, False],
    }

    # Confidence thresholds to try
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    schemes = {
        "1_to_1": "sing_label_data",
        "2_to_1": "sing_label_data2_1",
        "3_to_1": "sing_label_data3_1",
    }

    results_root = os.path.join(REPO_ROOT, "results", "single_label")
    os.makedirs(results_root, exist_ok=True)

    for scheme_name, subdir in schemes.items():
        print(f"\n=======================================")
        print(f"RUNNING SCHEME: {scheme_name} ({subdir})")
        print(f"=======================================")

        scheme_results = run_scheme(
            scheme_name,
            subdir,
            test_embeddings,
            test_labels,
            param_grid,
            thresholds,
        )

        out_path = os.path.join(results_root, f"{scheme_name}.json")
        with open(out_path, "w") as f:
            json.dump(scheme_results, f, indent=2)

        print(f"[INFO] Saved results for {scheme_name} to: {out_path}")

if __name__ == "__main__":
    main()

