FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# rasterio/pyproj/shapely ship manylinux wheels; libexpat and libgomp are the
# runtime pieces those wheels and lightgbm still link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=6 \
    CMD python -c "import json,urllib.request,sys; sys.exit(0 if json.load(urllib.request.urlopen('http://localhost:8000/api/v1/health',timeout=4))['status']=='ok' else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
