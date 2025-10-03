FROM python:3.10-slim

WORKDIR /app

# Set writable model cache
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

# Create the cache folder and give open permissions
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face exposes port 7860
EXPOSE 7860

# Run FastAPI with uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]