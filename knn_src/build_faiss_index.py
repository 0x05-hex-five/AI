import argparse
import pickle
from pathlib import Path

import pandas as pd
import faiss

"""
This script builds a FAISS FlatIP index from precomputed embeddings.
It loads embeddings and C-Code labels from a pickle file,
uses an Excel file to map each C-Code to an integer item_seq,
constructs a FAISS index for fast similarity search,
and saves both the index and the mapped item_seq labels for later use.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build FAISS FlatIP Index with item_seq labels')
    parser.add_argument('--emb',     default='embeddings/embeddings_dino_raw.pkl',
                        help='pickle file containing {"feats":…, "labels": [C-Code list]}')
    parser.add_argument('--index',   default='embeddings/knn_flatip.index',
                        help='where to write the FAISS index')
    parser.add_argument('--labels',  default='embeddings/labels.pkl',
                        help='where to write the mapped item_seq labels')
    parser.add_argument('--mapping', default='학습데이터_item_seq.xlsx',
                        help='Excel file with columns "C-Code" and "item_seq"')
    args = parser.parse_args()

    # load embeddings and original C-Code labels
    with open(args.emb, 'rb') as f:
        data = pickle.load(f)
    feats, ccode_labels = data['feats'], data['labels']
    assert feats.shape[0] == len(ccode_labels), "Number of embeddings and labels must match"

    # read mapping Excel and build lookup dict
    mapping_df = pd.read_excel(args.mapping)
    map_dict = dict(zip(
        mapping_df['C-Code'].astype(str),
        mapping_df['item_seq'].astype(int)
    ))

    # convert each C-Code to its integer item_seq
    item_seq_labels = []
    missing = set()
    for code in ccode_labels:
        key = str(code)
        if key in map_dict:
            item_seq_labels.append(map_dict[key])
        else:
            missing.add(key)
            item_seq_labels.append(-1)  # or choose a sentinel value
    if missing:
        raise KeyError(f"Missing mapping for C-Code(s): {missing}")

    # build FAISS FlatIP index
    dim = feats.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(feats)

    # save index and mapped labels
    Path(args.index).parent.mkdir(exist_ok=True, parents=True)
    faiss.write_index(index, args.index)

    Path(args.labels).parent.mkdir(exist_ok=True, parents=True)
    with open(args.labels, 'wb') as f:
        pickle.dump(item_seq_labels, f)

    print(f"Built FlatIP index with {index.ntotal} vectors -> {args.index}")
    print(f"Saved {len(item_seq_labels)} item_seq labels -> {args.labels}")
