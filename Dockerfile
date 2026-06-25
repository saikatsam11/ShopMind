FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install CPU-only torch from PyTorch index, then rest of deps
RUN pip install --no-cache-dir torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt --no-deps --ignore-requires-python || \
    pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY api.py .
COPY rag_pipeline.py .
COPY conversational_rag.py .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
