# -- base image --
# python 3.11-slim keeps the image small (~150MB vs ~900MB for full python)
# slim has everything we need for pandas/numpy/statsmodels
FROM python:3.11-slim AS base

# prevents python from buffering stdout/stderr
# (so docker logs show output in real-time, not when the buffer fills up)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# -- dependencies layer --
# copying requirements first means docker caches this layer
# so rebuilds only re-install if requirements.txt actually changed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -- application code --
COPY src/ ./src/
COPY api.py .
COPY data/ ./data/

# the API runs on port 8000
EXPOSE 8000

# health check so docker/k8s knows if the service is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# run with uvicorn - 4 workers for production, 1 for dev
# override workers via UVICORN_WORKERS env var
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
