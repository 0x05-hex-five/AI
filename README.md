# AI Module for Capstone Project

---

## 🔎 Overview

This project enables fast and accurate pill identification from images using self-supervised DINO embeddings and FAISS-backed KNN search. Designed for real-world pharmaceutical applications, it allows efficient retrieval and classification of pill types among 100,000+ images.

---

## 📂 Repository Structure

```text
AI/
├── data_preparation/          # download & unzip raw data, basic preprocessing
│   ├── add_item_seq.py
│   ├── check_zip_ccode.py
│   ├── extract_zips.py
│   ├── move_dataset.py
│   └── split_dataset.py
├── knn_src/                   # build and query FAISS index for KNN classification
│   ├── build_faiss_index.py
│   ├── extract_raw_embeddings.py
│   └── knn_utils.py
├── test/                      # test code
|   ├── predict_knn.py
│   └── test_gradio_knn.py
├── app.py                     # FastAPI app exposing /predict endpoint
├── Dockerfile                 # containerize the API
├── requirements.txt           # core Python dependencies
├── requirements_gradio.txt    # optional Gradio UI dependencies
├── pull_request_template.md   # PR template
├── .gitignore
├── .gitattributes
├── .dockerignore
└── README.md                  # this file
```

---

## 🚀 Getting Started

1. **Clone the repo**

   ```bash
   git clone https://github.com/0x05-hex-five/AI.git
   cd AI
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   .\.venv\Scripts\activate       # Windows PowerShell
   ```

3. **Install core dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install embeddings**

   ```bash
    huggingface-cli login
    git lfs install

    # Clone only embeddings folder
    git clone https://huggingface.co/danielee982/pill-project
    cd pill-project/embeddings
    ls
   ```
   Then, relocate "embeddings" folder into project root directory

5. *(Optional)* **Install Gradio UI dependencies**

   ```bash
   pip install -r requirements_gradio.txt
   ```

---

## ⚙️ Building the FAISS Index

```bash
python knn_src/build_faiss_index.py \
  --embeddings path/to/embeddings_dino_raw.pkl \
  --labels   path/to/labels.csv \
  --output   embeddings/knn_flatip.index
```

---

## 🌐 Running the API Server

Start the FastAPI server locally:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

* Visit the interactive docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📦 Docker

Build and run via Docker:

```bash
docker build -t ai-service:latest .
docker run -p 8000:8000 ai-service:latest
```
