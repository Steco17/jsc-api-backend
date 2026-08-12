FROM python:3.11-slim
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY data/languages.json data/languages.json

# The image intentionally does not bake multi-gigabyte model weights into an
# application layer. Mount model_ct2 and model_out/merged when the container is
# started. The unprivileged account owns data/ so contributions remain writable.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /srv/data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
