import numpy as np
import argparse
import os
import sys
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

LABELS = ["HD", "CV", "VO", "NONE"]


# ---------------------------------------------------------------------
# Utility to tee output to console AND file
# ---------------------------------------------------------------------
class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


# ---------------------------------------------------------------------
# Data loading functions
# ---------------------------------------------------------------------
def load_single_label_file(path):
    arr = np.load(path, allow_pickle=True)
    X = np.vstack(arr[:, 0])
    y_str = arr[:, 1]
    return X, y_str


def string_labels_to_int(y_str):
    mapping = {label: i for i, label in enumerate(LABELS)}
    return np.array([mapping[y] for y in y_str])


def convert_test_labels_to_multi_hot(y_test_raw):
    return y_test_raw.astype(int)


# ---------------------------------------------------------------------
# Train + tune
# ---------------------------------------------------------------------
def train_and_tune(X, y_int, threshold_values):
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y_int, test_size=0.22, random_state=42, stratify=y_int
    )

    model = OneVsRestClassifier(
        LinearSVC(C=10, tol=0.001, loss="squared_hinge", dual=False)
    )
    model.fit(X_train, y_train)

    dev_scores = model.decision_function(X_dev)

    best_thr = None
    best_f1 = -1

    for thr in threshold_values:
        preds = []
        for row in dev_scores:
            chosen = [LABELS[i] for i, s in enumerate(row) if s >= thr]
            if len(chosen) == 0:
                chosen = ["NONE"]
            preds.append(chosen)

        # true labels
        y_true_multi = np.zeros((len(y_dev), 3))
        for i, lab_int in enumerate(y_dev):
            if LABELS[lab_int] != "NONE":
                idx = LABELS.index(LABELS[lab_int])
                if idx < 3:
                    y_true_multi[i, idx] = 1

        # predicted labels
        y_pred_multi = np.zeros((len(preds), 3))
        for i, pred_list in enumerate(preds):
            for p in pred_list:
                if p in ["HD", "CV", "VO"]:
                    idx = LABELS.index(p)
                    y_pred_multi[i, idx] = 1

        f1_macro = f1_score(y_true_multi, y_pred_multi, average="macro")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_thr = thr

    print(f"\nBest threshold: {best_thr} (dev macro-F1={best_f1:.4f})")
    return best_thr


# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------
def evaluate_on_test(model, test_emb, test_labels, threshold):
    scores = model.decision_function(test_emb)

    preds = []
    for row in scores:
        chosen = [LABELS[i] for i, s in enumerate(row) if s >= threshold]
        if len(chosen) == 0:
            chosen = ["NONE"]
        preds.append(chosen)

    # build prediction vectors
    y_pred = np.zeros((len(preds), 3))
    for i, pred_list in enumerate(preds):
        for p in pred_list:
            if p in ["HD", "CV", "VO"]:
                idx = LABELS.index(p)
                y_pred[i, idx] = 1

    y_true = test_labels  # (N,3)

    # macro/micro F1
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")

    # per-label metrics
    print("\n=========== PER-LABEL METRICS ===========")
    for idx, label in enumerate(["HD", "CV", "VO"]):
        p = precision_score(y_true[:, idx], y_pred[:, idx], zero_division=0)
        r = recall_score(y_true[:, idx], y_pred[:, idx], zero_division=0)
        f = f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0)
        print(f"{label}:  Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")

    # NONE metrics
    none_true = (y_true.sum(axis=1) == 0).astype(int)
    none_pred = (y_pred.sum(axis=1) == 0).astype(int)

    p_none = precision_score(none_true, none_pred, zero_division=0)
    r_none = recall_score(none_true, none_pred, zero_division=0)
    f_none = f1_score(none_true, none_pred, zero_division=0)

    print(f"NONE: Precision={p_none:.4f} Recall={r_none:.4f} F1={f_none:.4f}")

    print("\n=========== FINAL TEST RESULTS ===========")
    print(f"Macro-F1 (HD, CV, VO): {f1_macro:.4f}")
    print(f"Micro-F1 (HD, CV, VO): {f1_micro:.4f}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to single-label npy file")
    args = parser.parse_args()

    # -------------------------------
    # SET RESULTS FILE
    # -------------------------------
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # UNIQUE OUTPUT NAME INCLUDING DIRECTORY
    dir_name = os.path.basename(os.path.dirname(args.file))
    base = os.path.basename(args.file).replace(".npy", "")
    out_file = os.path.join(results_dir, f"{dir_name}__single_label__{base}.txt")

    # Start tee logger
    Tee(out_file)

    print(f"\nSaving full output to: {out_file}\n")

    # -------------------------------
    # Load training data
    # -------------------------------
    X, y_str = load_single_label_file(args.file)
    y_int = string_labels_to_int(y_str)

    print(f"Loaded {len(X)} training rows from {args.file}")

    # -------------------------------
    # Tune threshold
    # -------------------------------
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    best_thr = train_and_tune(X, y_int, thresholds)

    # -------------------------------
    # Retrain on ALL data
    # -------------------------------
    model = OneVsRestClassifier(
        LinearSVC(C=10, tol=0.001, loss="squared_hinge", dual=False)
    )
    model.fit(X, y_int)

    # -------------------------------
    # Load test set
    # -------------------------------
    test_emb = np.load(os.path.join(repo_root, "gold_labels", "test_embeddings.npy"))
    test_labels_raw = np.load(os.path.join(repo_root, "gold_labels", "test_labels.npy"))
    test_labels_multi = convert_test_labels_to_multi_hot(test_labels_raw)

    # -------------------------------
    # FINAL evaluation
    # -------------------------------
    evaluate_on_test(model, test_emb, test_labels_multi, best_thr)

    print(f"\nResults saved to: {out_file}\n")


if __name__ == "__main__":
    main()

