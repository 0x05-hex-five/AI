# Use the same Python base image (3.9.22-slim) that your Docker Hub tag was built FROM
FROM python:3.9-slim

# Ensure UTF-8 locale
ENV LANG=C.UTF-8

# Work in /app
WORKDIR /app

# Update package lists (this matches the `apt-get update` layer you saw)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       # (add any extra system deps you need here, e.g. libgl1, build-essential, etc.) \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app and the KNN source code
COPY app.py ./app.py
COPY knn_src/ ./knn_src/

# Expose the same port
EXPOSE 8000

# Same entrypoint you saw in the history
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
