import os
import numpy as np
from itertools import product
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import argparse
import ast

# Get the repository root directory (parent of utils/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

def refine_labels(model, X_unlabeled, confidence_threshold):
    decision_function = model.decision_function(X_unlabeled)
    preds = (decision_function >= confidence_threshold).astype(int)

    # if a sample has no labels above threshold, mark "None"
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    none_mask = preds.sum(axis=1) == 0
    if none_mask.any():
        none_col = np.zeros((preds.shape[0], 1), dtype=int)
        none_col[none_mask, 0] = 1
        preds = np.hstack([preds, none_col])
    return preds

def save_confidence_scores_to_files(models, test_embeddings, label_names=None, output_dir=None):
    """Save confidence (decision function) scores for each label classifier."""
    if output_dir is None:
        output_dir = os.path.join(REPO_ROOT, 'logs', 'confidence_scores')
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(REPO_ROOT, 'logs', output_dir)
    
    if hasattr(models, "estimators_"):
        n_labels = len(models.estimators_)
    else:
        n_labels = 1

    if label_names is None or len(label_names) != n_labels:
        label_names = [f"Label_{i}" for i in range(n_labels)]

    os.makedirs(output_dir, exist_ok=True)
    for i, model in enumerate(models.estimators_):
        scores = model.decision_function(test_embeddings)
        filename = os.path.join(output_dir, f"confidence_{label_names[i]}.txt")
        with open(filename, "w") as f:
            f.write(f"Confidence scores for label: {label_names[i]}\n")
            for j, score in enumerate(scores):
                f.write(f"Sample {j}: {score:.4f}\n")
    print(f"[INFO] Confidence scores saved in: {os.path.abspath(output_dir)}")

def load_methods(setting, data_dir):
    methods = []
    if not os.path.exists(data_dir):
        print(f"[WARNING] Data directory {data_dir} does not exist")
        return methods
    for file in os.listdir(data_dir):
        if file.endswith("_single_label.npy"):
            prefix = file.replace("_single_label.npy", "")
            filepath = os.path.join(data_dir, file)
            methods.append((prefix, filepath, None))
    return methods

def main(args):
    # Load test embeddings and labels from gold_labels/
    test_emb_file = os.path.join(REPO_ROOT, 'gold_labels', 'test_embeddings.npy')
    test_lbl_file = os.path.join(REPO_ROOT, 'gold_labels', 'test_labels.npy')
    test_embeddings = np.load(test_emb_file)
    test_labels = np.load(test_lbl_file)
    
    # Resolve data_dir path
    if not os.path.isabs(args.data_dir):
        data_dir = os.path.join(REPO_ROOT, 'data', args.data_dir)
    else:
        data_dir = args.data_dir
    
    # Handle file argument if provided
    if args.file:
        # If file is provided, use it directly
        if os.path.isabs(args.file):
            file_path = args.file
        else:
            # Assume relative paths are relative to data/ directory
            file_path = os.path.join(REPO_ROOT, args.file)
        methods = [(os.path.splitext(os.path.basename(args.file))[0].replace('_single_label', ''), file_path, None)]
    else:
        methods = load_methods(args.setting, data_dir)

    label_scheme = {
        "HD": [1, 0, 0],
        "CV": [0, 1, 0],
        "VO": [0, 0, 1],
        "None": [0, 0, 0]
    }

    ### FIX: Define label sets for each mode
    label_names_single = ["HD", "CV", "VO", "None"]
    label_names_multiclass = [f"Class_{i}" for i in range(len(np.unique(test_labels)))]

    for method_name, emb_file, label_file in methods:
        print(f"[INFO] Processing method: {method_name}")

        # --- Load embeddings and labels ---
        if method_name == "baseline" or args.setting == "baseline":
            # Baseline data should be in gold_labels/
            if args.baseline_data_dir:
                if os.path.isabs(args.baseline_data_dir):
                    baseline_dir = args.baseline_data_dir
                else:
                    baseline_dir = os.path.join(REPO_ROOT, args.baseline_data_dir)
            else:
                baseline_dir = os.path.join(REPO_ROOT, 'gold_labels')
            train_embeddings = np.load(os.path.join(baseline_dir, 'train_embeddings.npy'), allow_pickle=True)
            train_labels = np.load(os.path.join(baseline_dir, 'train_labels.npy'), allow_pickle=True)
        else:
            data = np.load(emb_file, allow_pickle=True)
            label_map = {"HD": 0, "CV": 1, "VO": 2, "NONE": 3, "None": 3}

            if isinstance(data, np.ndarray) and data.dtype == object:
                if all(isinstance(x, (tuple, list, np.ndarray)) and len(x) == 2 for x in data):
                    train_embeddings = np.array([np.array(x[0], dtype=float) for x in data])
                    train_labels = np.array([
                        label_map[x[1]] if isinstance(x[1], str) else int(x[1])
                        for x in data
                    ])
                else:
                    try:
                        train_embeddings, train_labels = data
                        train_embeddings = np.array(train_embeddings, dtype=float)
                        train_labels = np.array([
                            label_map[x] if isinstance(x, str) else int(x)
                            for x in train_labels
                        ])
                    except Exception:
                        raise ValueError(f"Could not parse object array in {emb_file}")
            elif isinstance(data, tuple) and len(data) == 2:
                train_embeddings = np.array(data[0], dtype=float)
                train_labels = np.array([
                    label_map[x] if isinstance(x, str) else int(x)
                    for x in data[1]
                ])
            elif isinstance(data, (list, np.ndarray)) and len(data) > 0 and \
                 isinstance(data[0], (tuple, list)) and len(data[0]) == 2:
                train_embeddings = np.array([np.array(x[0], dtype=float) for x in data])
                train_labels = np.array([
                    label_map[x[1]] if isinstance(x[1], str) else int(x[1])
                    for x in data
                ])
            else:
                raise ValueError(f"Unrecognized structure in {emb_file}, and no separate label file provided.")

        y = np.array(train_labels)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if args.split_dev:
            train_emb, dev_emb, train_lbls, dev_lbls = train_test_split(
                train_embeddings, y, test_size=0.222, random_state=42
            )
            print(f"[INFO] Training samples: {len(train_emb)}")
            print(f"[INFO] Dev samples: {len(dev_emb)}")
        else:
            train_emb, train_lbls = train_embeddings, y
            dev_emb, dev_lbls = None, None
            print(f"[INFO] Training samples: {len(train_emb)}")

        print(f"[INFO] Test samples: {len(test_embeddings)}")

        fixed_params = {'C': 10, 'tol': 0.001, 'loss': 'squared_hinge', 'penalty': 'l2', 'dual': False}

        # === Training modes ===
        if dev_emb is not None and args.use_confidence_dev:
            print("[INFO] Using dev set to select best confidence threshold (fixed hyperparameters)")
            best_score = -1
            best_threshold = 0.7
            base_model = LinearSVC(**fixed_params, max_iter=10000, random_state=42)
            ovr_model = OneVsRestClassifier(base_model)
            ovr_model.fit(train_emb, train_lbls)

            for threshold in np.arange(0.3, 0.91, 0.05):
                preds_dev = (ovr_model.decision_function(dev_emb) >= threshold).astype(int)
                macro_f1 = f1_score(dev_lbls, preds_dev, average='macro', zero_division=0)
                if macro_f1 > best_score:
                    best_score = macro_f1
                    best_threshold = threshold
            print(f"[INFO] Best confidence threshold on dev set: {best_threshold:.2f}")
            combined_emb = np.vstack([train_emb, dev_emb])
            combined_lbls = np.vstack([train_lbls, dev_lbls])
            best_params = fixed_params

        elif dev_emb is not None and args.hyperparam_search:
            print("[INFO] Performing hyperparameter grid search on dev set")
            best_score = -1
            best_params = fixed_params
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'tol': [1e-4, 1e-3, 1e-2],
                'loss': ['hinge', 'squared_hinge'],
                'penalty': ['l2'],
                'dual': [True, False]
            }
            for combo in product(*param_grid.values()):
                params = dict(zip(param_grid.keys(), combo))
                try:
                    base_model = LinearSVC(**params, max_iter=10000, random_state=42)
                    ovr_model = OneVsRestClassifier(base_model)
                    ovr_model.fit(train_emb, train_lbls)
                    preds_dev = ovr_model.predict(dev_emb)
                    macro_f1 = f1_score(dev_lbls, preds_dev, average='macro')
                    if macro_f1 > best_score:
                        best_score = macro_f1
                        best_params = params
                except Exception as e:
                    print(f"[WARN] Skipping params {params}: {str(e)}")
            print(f"[INFO] Best params from grid search: {best_params}")
            combined_emb = np.vstack([train_emb, dev_emb])
            combined_lbls = np.vstack([train_lbls, dev_lbls])
        else:
            print("[INFO] No dev set, training with default or specified params")
            combined_emb = train_emb
            combined_lbls = train_lbls
            if args.model_params:
                best_params = ast.literal_eval(args.model_params)
            else:
                best_params = fixed_params
            best_threshold = 0.7

        base_model = LinearSVC(**best_params, max_iter=10000, random_state=42)
        final_model = OneVsRestClassifier(base_model)
        final_model.fit(combined_emb, combined_lbls)
        preds_test = final_model.predict(test_embeddings)

        if preds_test.ndim == 1:
            preds_test = preds_test.reshape(-1, 1)
        if test_labels.ndim == 1:
            test_labels = test_labels.reshape(-1, 1)

        if preds_test.shape[1] != test_labels.shape[1]:
            min_labels = min(preds_test.shape[1], test_labels.shape[1])
            preds_test = preds_test[:, :min_labels]
            test_labels = test_labels[:, :min_labels]

        ### FIX: Choose correct label names per setting
        if args.setting == "multiclass":
            label_names_all = label_names_multiclass
        else:
            label_names_all = label_names_single

        if args.confidence:
            save_confidence_scores_to_files(final_model, test_embeddings, label_names_all)

        # === Evaluation ===
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
        label_names = label_names_all

        best_macro_f1 = -1
        best_results = None
        best_thresh = None

        for thr in thresholds:
            preds_thr = refine_labels(final_model, test_embeddings, thr)
            min_labels = min(preds_thr.shape[1], test_labels.shape[1])
            preds_thr = preds_thr[:, :min_labels]
            gold = test_labels[:, :min_labels]

            label_metrics = []
            for i in range(gold.shape[1]):
                tp = np.sum((preds_thr[:, i] == 1) & (gold[:, i] == 1))
                fp = np.sum((preds_thr[:, i] == 1) & (gold[:, i] == 0))
                fn = np.sum((preds_thr[:, i] == 0) & (gold[:, i] == 1))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                label_metrics.append((precision, recall, f1))

            avg_prec = np.mean([x[0] for x in label_metrics])
            avg_recall = np.mean([x[1] for x in label_metrics])
            avg_f1 = np.mean([x[2] for x in label_metrics])
            micro_f1 = f1_score(gold, preds_thr, average="micro", zero_division=0)
            macro_f1 = f1_score(gold, preds_thr, average="macro", zero_division=0)

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_results = {
                    "threshold": thr,
                    "per_label": label_metrics,
                    "avg": (avg_prec, avg_recall, avg_f1),
                    "micro_f1": micro_f1,
                    "macro_f1": macro_f1
                }
                best_thresh = thr

        print(f"\n[THRESHOLD] Best = {best_thresh}")
        print("\nPer-label metrics:")
        for i, (p, r, f1) in enumerate(best_results["per_label"]):
            label = label_names[i] if i < len(label_names) else f"Label_{i}"
            scheme = label_scheme.get(label, [0, 0, 0])
            print(f"{label}: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}, Scheme={scheme}")

        ap, ar, af = best_results["avg"]
        print(f"\n[AVG] Precision={ap:.4f}, Recall={ar:.4f}, F1={af:.4f}")
        print("\n[SKLEARN] Evaluation:")
        print(f"Micro F1: {best_results['micro_f1']:.4f}")
        print(f"Macro F1: {best_results['macro_f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--confidence", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--labels", type=int, default=4)
    parser.add_argument("--baseline_data_dir", type=str)
    parser.add_argument("--split_dev", action="store_true")
    parser.add_argument("--setting", choices=["binary", "multiclass", "baseline", "single"], required=True)
    parser.add_argument("--hyperparam_search", action="store_true")
    parser.add_argument("--use_confidence_dev", action="store_true")
    parser.add_argument("--model_params", type=str, default=None)
    parser.add_argument('--file', type=str, default=None, help="Path to a .npy file to load directly")
    args = parser.parse_args()

    if args.data_dir == "all":
        for directory in ["resampled_data", "resampled_data2_1", "resampled_data3_1",
                          "sing_label_data", "sing_label_data2_1", "sing_label_data3_1"]:
            print(f"\n--- Processing directory: {directory} ---")
            args.data_dir = directory
            # For baseline, use gold_labels directory
            if args.setting == "baseline":
                args.baseline_data_dir = "gold_labels"
            else:
                args.baseline_data_dir = None
            try:
                main(args)
            except Exception as e:
                print(f"[ERROR] Failed to process {directory}: {str(e)}")
    else:
        if args.baseline_data_dir and not os.path.isabs(args.baseline_data_dir):
            # If baseline_data_dir is relative, it should be relative to gold_labels
            if args.baseline_data_dir == "baseline_data" or "gold" not in args.baseline_data_dir:
                args.baseline_data_dir = os.path.join(REPO_ROOT, 'gold_labels')
        main(args)

