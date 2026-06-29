# =============================================================================
# Dockerfile — LocalLLM-DataForge
# =============================================================================
# Multi-etapa con dependencias primero para aprovechar caching de capas.
# =============================================================================

FROM python:3.11-slim AS builder

WORKDIR /build

# Solo los archivos de dependencias para cachearlas como capa independiente
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir .

# ─────────────────────────────────────────────────────────────────────────────
# Imagen final — más chica, solo lo necesario para correr
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Etiquetas
LABEL org.opencontainers.image.title="LocalLLM-DataForge" \
      org.opencontainers.image.description="DataFrame-based pipeline for user story task decomposition using local LLMs" \
      org.opencontainers.image.version="0.4.0" \
      org.opencontainers.image.authors="Angel L. Valdés"

# Copiar Python y dependencias instaladas desde builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar el código fuente del proyecto
COPY pyproject.toml README.md ./
COPY scripts/ ./scripts/
COPY examples/ ./examples/
COPY src/ ./src/
COPY data/ ./data/

# Crear directorios para outputs y caché (como volúmenes montables)
RUN mkdir -p /app/data/outputs /app/.cache/dataforge

# Exponer puerto si hiciera falta (la app no es servidor, solo documentativo)
# EXPOSE 8000

# Punto de entrada: shell para comandos ad-hoc
CMD ["python"]
