import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import argparse
import os
import sys

# Get the repository root directory (parent of utils/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract BERT embeddings from TSV file')

parser.add_argument('--output_labels', type=str, default=None, help='Output labels file path (relative to gold_labels/)')
args = parser.parse_args()

# Set default input file if not provided
if args.input is None:
    INPUT_FILE = os.path.join(REPO_ROOT, 'data', 'ghc_train.tsv')
else:
    # If input is absolute, use it; otherwise relative to data/
    if os.path.isabs(args.input):
        INPUT_FILE = args.input
    else:
        INPUT_FILE = os.path.join(REPO_ROOT, 'data', args.input)

# Set paths for output files
if args.output_embeddings:
    if os.path.isabs(args.output_embeddings):
        OUTPUT_EMBEDDINGS = args.output_embeddings
    else:
        OUTPUT_EMBEDDINGS = os.path.join(REPO_ROOT, 'gold_labels', args.output_embeddings)
else:
    # Auto-generate output filenames based on input filename
    base_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    if 'train' in base_name.lower():
        OUTPUT_EMBEDDINGS = os.path.join(REPO_ROOT, 'gold_labels', 'train_embeddings.npy')
        OUTPUT_LABELS = os.path.join(REPO_ROOT, 'gold_labels', 'train_labels.npy')
    elif 'test' in base_name.lower():
        OUTPUT_EMBEDDINGS = os.path.join(REPO_ROOT, 'gold_labels', 'test_embeddings.npy')
        OUTPUT_LABELS = os.path.join(REPO_ROOT, 'gold_labels', 'test_labels.npy')
    else:
        OUTPUT_EMBEDDINGS = os.path.join(REPO_ROOT, 'gold_labels', f"{base_name}_embeddings.npy")
        OUTPUT_LABELS = os.path.join(REPO_ROOT, 'gold_labels', f"{base_name}_labels.npy")

if args.output_labels:
    if os.path.isabs(args.output_labels):
        OUTPUT_LABELS = args.output_labels
    else:
        OUTPUT_LABELS = os.path.join(REPO_ROOT, 'gold_labels', args.output_labels)

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_LABELS), exist_ok=True)

# Load the data
print("Loading dataset...")
data = pd.read_csv(INPUT_FILE, sep='\t')

# Ensure the TSV file has the expected structure
if 'text' not in data.columns or not {'hd', 'cv', 'vo'}.issubset(data.columns):
    raise ValueError("The input file must contain 'text', 'hd', 'cv', and 'vo' columns.")

# Separate text and labels
texts = data['text']
labels = data[['hd', 'cv', 'vo']].values

# Load pre-trained BERT tokenizer and model
print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to compute embeddings
def compute_embeddings(texts, batch_size=32):
    embeddings = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computations for efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(list(batch_texts), return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            # Use mean pooling of token embeddings
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Generate embeddings
print("Generating embeddings...")
embeddings = compute_embeddings(texts)

# Save embeddings and labels
print("Saving embeddings and labels...")
np.save(OUTPUT_EMBEDDINGS, embeddings)
np.save(OUTPUT_LABELS, labels)

print(f"Embeddings saved to {OUTPUT_EMBEDDINGS}")
print(f"Labels saved to {OUTPUT_LABELS}")

