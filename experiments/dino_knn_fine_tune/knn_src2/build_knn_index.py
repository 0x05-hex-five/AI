import argparse
import pickle
import faiss
from pathlib import Path


def build_index(emb_path: str, idx_path: str, lbl_path: str):
    # Load embeddings (dict or tuple)
    with open(emb_path, 'rb') as f:
        data = pickle.load(f)
    feats = data['feats']
    labels = data['labels']

    # L2-normalize for cosine similarity
    faiss.normalize_L2(feats)

    # Build FAISS index (inner product = cosine on normalized vectors)
    dim = feats.shape[1]
    index = faiss.IndexFlatIP(dim)
    if isinstance(index, faiss.IndexIVFPQ):
        index.train(feats)
    index.add(feats)

    # Save index and labels
    Path(idx_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, idx_path)
    Path(lbl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_path, 'wb') as f:
        pickle.dump(labels, f)

    print(f"Built FAISS index: {idx_path} ({feats.shape[0]} vectors)")
    print(f"Saved labels file:   {lbl_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build FAISS k-NN index and labels from embeddings"
    )
    parser.add_argument(
        '-e', '--embeddings', default="embeddings2/embeddings_fine_tuned.pkl",
        help='Path to embeddings.pkl (dict with "feats"/"labels" or tuple)'
    )
    parser.add_argument(
        '-i', '--index', default="embeddings2/knn_cosine.index",
        help='Output path for FAISS index (.index)'
    )
    parser.add_argument(
        '-l', '--labels', default="embeddings2/labels.pkl",
        help='Output path for labels.pkl'
    )
    args = parser.parse_args()
    build_index(args.embeddings, args.index, args.labels)