FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CPU-only torch — the CUDA build is 2 GB+ and unnecessary on App Platform
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the NER model at build time so cold starts are fast
RUN python -c "from transformers import pipeline; pipeline('ner', model='dslim/bert-base-NER', aggregation_strategy='simple')"

COPY . .

EXPOSE 8080

# 1 worker: NER model is heavy, multiple workers would each load a full copy
# timeout 120: OCR on scanned PDFs can take 30-60 s
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "app:app"]
