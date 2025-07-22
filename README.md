# 🧩 Pill Image Retrieval & Search Pipeline
This repository implements a robust backend for real-time pill image search and retrieval using DINO-ViT embeddings and FAISS KNN search. It is designed as a modular API component within a larger end-to-end pill identification service.

## 📂 Pipeline Overview
1. Data Preparation (data_preparation/)
   - Extracts and organizes raw pill image datasets.
   - Performs train/val/test splits and data integrity checks.

2. Reference Embedding Extraction (knn_src/extract_raw_embeddings.py)
   - Uses a DINO-ViT model to extract embeddings for all reference pill images.
   - Stores output as .pkl embedding files and corresponding label CSVs.

3. FAISS Index Construction (knn_src/build_faiss_index.py)
   - Loads embeddings and labels to build a FAISS index (FlatIP).
   - Saves as knn_flatip.index for efficient similarity search.

4. KNN Search & Inference (knn_src/knn_utils.py, test/predict_knn.py)
   - For each query (user-uploaded) image:
     - Extracts its DINO embedding.
     - Searches the FAISS index for the top-k most similar reference pills.
     - Returns pill code, name, and similarity scores.

5. API Service (app.py)
   - FastAPI server exposes a `/predict` endpoint.
   - Accepts image uploads and returns top pill candidates as JSON.
   - Includes a Gradio-based demo UI (`test/test_gradio_knn.py`) for interactive local testing.

6. Docker & Deployment
   - Dockerfile and requirements.txt are provided for easy containerization and deployment.

## 🔗 End-to-End Service Flow
1. The user uploads a photo containing a pill.
2. The application provides an interface for the user to crop the image to focus on the individual pill.
3. The cropped pill image is sent to this module, which extracts its DINO embedding, performs a KNN search with FAISS, and returns the top matching reference pills.
4. The frontend displays the candidate pill names, codes, and similarity scores to the user.

## 🏷️ Key Files & Roles
| File/Folder                         | Description                                      |
| ----------------------------------- | ------------------------------------------------ |
| `data_preparation/`                 | Data download, extraction, splitting, cleaning   |
| `knn_src/extract_raw_embeddings.py` | DINO-ViT feature extraction for reference images |
| `knn_src/build_faiss_index.py`      | FAISS index creation                             |
| `knn_src/knn_utils.py`              | Embedding, KNN search, result formatting         |
| `test/predict_knn.py`               | Local/batch prediction tests                     |
| `app.py`                            | FastAPI inference server                         |
| `test/test_gradio_knn.py`           | Gradio demo UI                                   |
| `Dockerfile`                        | Containerization                                 |
| `requirements.txt`                  | Dependencies                                     |

