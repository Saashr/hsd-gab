import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import argparse
import os

# Get repository root (parent of utils/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

###############################################
# ARGUMENT PARSER
###############################################
parser = argparse.ArgumentParser(description='Extract BERT embeddings from TSV file')

parser.add_argument('--input', type=str, required=True,
                    help='Input TSV filename (e.g., ghc_train.tsv) or absolute path')

parser.add_argument('--output_embeddings', type=str, required=True,
                    help='Output embeddings filename (.npy)')

parser.add_argument('--output_labels', type=str, required=True,
                    help='Output labels filename (.npy)')

args = parser.parse_args()

###############################################
# FIXED PATH HANDLING
###############################################
# Resolve input file
if os.path.isabs(args.input):
    INPUT_FILE = args.input
else:
    INPUT_FILE = os.path.join(REPO_ROOT, "data", args.input)

# Resolve output paths
def resolve_output(path):
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, "gold_labels", path)

OUTPUT_EMBEDDINGS = resolve_output(args.output_embeddings)
OUTPUT_LABELS = resolve_output(args.output_labels)

os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_LABELS), exist_ok=True)

###############################################
# LOAD DATA
###############################################
print(f"Loading dataset from {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

data = pd.read_csv(INPUT_FILE, sep='\t')

required_cols = {'text', 'hd', 'cv', 'vo'}
if not required_cols.issubset(data.columns):
    raise ValueError(f"Input TSV must contain columns: {required_cols}")

texts = data['text']
labels = data[['hd', 'cv', 'vo']].values

###############################################
# LOAD BERT MODEL
###############################################
print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

###############################################
# EMBEDDING FUNCTION
###############################################
def compute_embeddings(texts, batch_size=32):
    embeddings = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(pooled)

    return np.vstack(embeddings)

###############################################
# COMPUTE + SAVE
###############################################
print("Generating embeddings...")
embeddings = compute_embeddings(texts)

print(f"Saving embeddings to: {OUTPUT_EMBEDDINGS}")
np.save(OUTPUT_EMBEDDINGS, embeddings)

print(f"Saving labels to: {OUTPUT_LABELS}")
np.save(OUTPUT_LABELS, labels)

print("Done.")

