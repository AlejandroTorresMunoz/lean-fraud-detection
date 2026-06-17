# Minimal image for the FastAPI scorer.
FROM python:3.11-slim

WORKDIR /app

# Install only what's needed to serve (keep the image lean).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "lean_fraud.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
