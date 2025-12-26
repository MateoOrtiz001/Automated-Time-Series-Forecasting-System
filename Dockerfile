# =============================================================================
# Dockerfile - Sistema Automatizado de Predicción de Inflación
# =============================================================================
# Build:  docker build -t inflation-forecast .
# Run:    docker run -v $(pwd)/data:/app/data -v $(pwd)/misc:/app/misc inflation-forecast
# =============================================================================

FROM python:3.11-slim

# Metadatos
LABEL maintainer="Automated Time Series Forecasting System"
LABEL description="Sistema de predicción de inflación con TFT (Temporal Fusion Transformer)"
LABEL version="1.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY misc/ ./misc/
COPY feature_analysis.py .
COPY test.py .
COPY test2.py .

# Crear directorios necesarios
RUN mkdir -p data/raw/banrep/suameca \
             data/raw/external \
             data/proc \
             misc/models \
             misc/results \
             misc/logs \
             models \
             results

# Copiar modelos si existen (opcional, se pueden montar como volumen)
COPY models/*.keras ./models/ 2>/dev/null || true
COPY misc/models/*.keras ./misc/models/ 2>/dev/null || true

# Puerto para posible API futura
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tensorflow; print('OK')" || exit 1

# Comando por defecto: ejecutar pipeline completo
ENTRYPOINT ["python", "src/pipeline/orchestrator.py"]
CMD ["--mode", "full"]
