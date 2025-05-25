import argparse, os
import pickle
import numpy as np
import faiss
from PIL import Image
import torch
from collections import OrderedDict

from knn_src.knn_utils import get_transform

"""
Script for predicting image classes using a FAISS K-NN index and a DINO backbone.
Args:
    image (str): Path to the input image or directory of images. Defaults to 'test_data'.
    -k (int): Number of top neighbors to retrieve. Defaults to 100.
    --index (str): Path to the FAISS index file. Defaults to 'embeddings/knn_flatip.index'.
    --labels (str): Path to the pickle file containing class labels. Defaults to 'embeddings/labels.pkl'.
    --size (int): Image size for preprocessing. Defaults to 224.
Workflow:
    1. Loads a FAISS index and associated class labels.
    2. Loads a DINO ViT-S/16 backbone for feature extraction.
    3. For each image in the input directory:
        a. Extracts features using the backbone.
        b. Searches the FAISS index for top-k nearest neighbors.
        c. Aggregates the highest similarity per class.
        d. Sorts and selects the top-5 classes by similarity.
        e. Checks if the ground-truth class is in the top-5 predictions.
    4. Prints the number of images where the ground-truth class is in the top-5 predictions.
Note:
    - Assumes image filenames encode the ground-truth class as the integer prefix before the first dot.
    - Requires 'knn_src.knn_utils.get_transform' for image preprocessing.
    - Requires FAISS, PyTorch, and DINO model weights.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with FAISS K-NN')
    parser.add_argument('image', nargs='?', default='test_data', help='Input image path')
    parser.add_argument('-k',    type=int,   default=100, help='Top-k neighbors')
    parser.add_argument('--index',  default='embeddings/knn_flatip.index')
    parser.add_argument('--labels', default='embeddings/labels.pkl')
    parser.add_argument('--size',   type=int,   default=224)
    args = parser.parse_args()

    # Load FAISS index and labels
    index = faiss.read_index(args.index)
    with open(args.labels, 'rb') as f:
        class_labels = pickle.load(f)

    # Load backbone
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval()
    transform = get_transform(args.size)

    top5_cnt = 0
    for img_name in os.listdir(args.image):
        print(f"file name: {img_name}")
        img_path = os.path.join(args.image, img_name)
        # Embed query
        img = Image.open(img_path).convert('RGB')
        x   = transform(img).unsqueeze(0)
        with torch.no_grad():
            feats = model(x).squeeze(0)
            feats = feats / feats.norm()
        vector = feats.cpu().numpy().astype('float32').reshape(1, -1)

        # Search top-k
        D, I = index.search(vector, args.k)
        sims = D[0] # top-k similarity
        idxs = I[0] # top-k index

        # collect the highest similarity per class
        unique = OrderedDict()
        top_list = []  # list of (idx, similarity)
        for sim, idx in sorted(zip(sims, idxs), key=lambda x: -x[0]):
            class_name = class_labels[int(idx)]

            # update if class not seen or found with higher similarity
            if class_name not in unique or sim > unique[class_name]:
                unique[class_name] = float(sim)

        # sort top-5        
        top_list = sorted(unique.items(), key=lambda x: x[1], reverse=True)[:5] # [(class_name, sim), ...]
        print(top_list)

        gt = int(img_name.split('.')[0])
        pred_classes = [cls for cls, sim in top_list]

        if gt in pred_classes:
            top5_cnt += 1

    print(f"{top5_cnt} / {len(os.listdir(args.image))}")