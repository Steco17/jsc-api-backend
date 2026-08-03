FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
# Copy your converted model + tokenizer into the image (or mount as volumes)
# COPY model_ct2/ model_ct2/
# COPY model_out/merged/ model_out/merged/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
