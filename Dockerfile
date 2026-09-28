# syntax=docker/dockerfile:1.7
FROM node:22-bookworm-slim AS web-build
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    MODEL_PATH=models/ppo_cr_best.zip

WORKDIR /app
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY requirements-prod.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" \
    && pip install --no-cache-dir -r requirements-prod.txt

COPY --chown=app:app . ./
COPY --from=web-build --chown=app:app /app/web/dist ./web/dist
RUN pip install --no-cache-dir --no-deps .

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/api/health', timeout=4)"

CMD ["sh", "-c", "python -m cr_rl.deploy.prepare_model && exec uvicorn cr_rl.server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
