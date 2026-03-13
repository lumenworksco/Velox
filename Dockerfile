FROM python:3.13-slim

# Create non-root user
RUN groupadd -r botuser && useradd -r -g botuser -m botuser

WORKDIR /app

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories for persistent data
RUN mkdir -p data logs && chown -R botuser:botuser /app

USER botuser

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8080/health'); assert r.status_code == 200"

CMD ["python", "main.py"]
