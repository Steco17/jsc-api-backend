FROM python:3.11-slim-trixie@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall --yes wheel

COPY app/ app/
COPY scripts/seed_admin.py scripts/seed_admin.py
COPY scripts/start_server.py scripts/start_server.py
COPY data/languages.json config/languages.json

# The image intentionally does not bake multi-gigabyte model weights into an
# application layer. Mount model_ct2 and model_out/merged when the container is
# started. The unprivileged account owns data/ so contributions remain writable.
RUN mkdir -p /srv/data \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /srv/data \
    && chmod 0700 /srv/data
USER 10001:10001

ENV DATA_DIR=/srv/data \
    DATABASE_PATH=/srv/data/jsc.sqlite3 \
    LANGUAGES_FILE=/srv/config/languages.json

EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.getenv('PORT', '8000')}/ready\", timeout=3)"]
CMD ["python", "-m", "scripts.start_server"]
