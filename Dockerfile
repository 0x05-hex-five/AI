# --------------------------------------------------
FROM python:3.9-slim
# system dependencies for OpenCV (libGL and etc.)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir huggingface-hub

# HF token
ARG HF_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}

# Download the model
RUN mkdir -p weights && \
    python - <<EOF
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="danielee982/pill-project",
    local_dir="weights",
    resume_download=True,
)
EOF

# debugging: print the contents of the weights directory
RUN echo "[DEBUG] /app/weights contents:" && ls -l /app/weights

# copy the rest of the code
COPY src/ src/
COPY class_mapping.xlsx .

# start the app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
# ---------------------------------------------------