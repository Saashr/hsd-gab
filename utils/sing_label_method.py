import numpy as np
import argparse
import os
import json

def process_embeddings(embeddings, labels, tag_columns):
    assert embeddings.shape[0] == labels.shape[0], "Embeddings and labels must have same number of rows."

    output = []
    num_duplicates = 0
    num_none = 0

    for i in range(embeddings.shape[0]):
        emb = embeddings[i]
        label_row = labels[i]

        tags_added = 0
        for j, tag in enumerate(label_row):
            if int(tag) == 1:
                output.append([emb, tag_columns[j]])
                tags_added += 1

        if tags_added > 1:
            num_duplicates += tags_added - 1
        if tags_added == 0:
            output.append([emb, "NONE"])
            num_none += 1

    return np.array(output, dtype=object), num_duplicates, num_none

def main():
    parser = argparse.ArgumentParser(description="Convert multi-label embeddings to single-label format.")
    parser.add_argument('--train_embeddings', required=True, help="Path to .npy file of embeddings")
    parser.add_argument('--train_labels', required=True, help="Path to .npy file of labels")
    parser.add_argument('--output', required=True, help="Path to output .npy file")
    parser.add_argument('--tag_columns', type=str, default='["HD", "CV", "VO"]',
                        help='JSON-style list of tag column names (e.g. \'["HD", "CV", "VO"]\')')

    args = parser.parse_args()

    # Load data
    embeddings = np.load(args.train_embeddings)
    labels = np.load(args.train_labels)
    tag_columns = json.loads(args.tag_columns)

    # Process
    output_array, num_duplicates, num_none = process_embeddings(embeddings, labels, tag_columns)

    # Save
    np.save(args.output, output_array)

    # Log
    print(f"Saved {len(output_array)} single-label rows to '{args.output}'.")
    print(f"Added {num_duplicates} duplicated rows due to multi-label entries.")
    print(f"Found {num_none} 'NONE' label rows.")

if __name__ == "__main__":
    main()

