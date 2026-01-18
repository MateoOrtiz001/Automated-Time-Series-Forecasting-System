# =============================================================================
# Dockerfile - Sistema Automatizado de Predicción de Inflación
# =============================================================================
# Build:  docker build -t inflation-forecast .
# Run:    docker run -v $(pwd)/data:/app/data -v $(pwd)/misc:/app/misc inflation-forecast
#
# Modos de ejecución:
#   - Pipeline mensual: docker run ... inflation-forecast pipeline
#   - Tests:            docker run ... inflation-forecast test
#   - Dashboard:        docker run -p 8501:8501 ... inflation-forecast web
#   - Shell:            docker run -it ... inflation-forecast shell
# =============================================================================

FROM python:3.11-slim

# Metadatos
LABEL maintainer="Automated Time Series Forecasting System"
LABEL description="Sistema de predicción de inflación con TFT (Temporal Fusion Transformer)"
LABEL version="2.0"

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
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install pytest pytest-cov

# Copiar código fuente
COPY src/ ./src/

COPY tests/ ./tests/

# Crear directorios necesarios (se sobreescribirán con volúmenes montados)
RUN mkdir -p data/raw/banrep/suameca \
             data/raw/external \
             data/proc \
             misc/models \
             misc/results \
             misc/logs \
             models \
             results

# Puerto para dashboard Streamlit
EXPOSE 8501

# Healthcheck básico (verifica imports críticos)
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD python -c "import tensorflow; import pandas; import numpy; print('OK')" || exit 1

# Script de entrada flexible
COPY <<'EOF' /app/entrypoint.sh
#!/bin/bash
set -e

MODE="${1:-pipeline}"

case "$MODE" in
    pipeline)
        echo " Running monthly pipeline..."
        shift || true
        exec python -m src.pipeline.monthly_pipeline "$@"
        ;;
    orchestrator)
        echo " Running orchestrator..."
        shift || true
        exec python -m src.pipeline.orchestrator "$@"
        ;;
    test)
        echo " Running tests..."
        shift || true
        exec python -m pytest tests/ -v --tb=short "$@"
        ;;
    test-quick)
        echo " Running quick import tests..."
        exec python -c "
import sys
print('Testing imports...')
try:
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    from src.model.model import TFTModel
    from src.pipeline.core import run_pipeline
    print(' All imports successful')
    print(f'   TensorFlow: {tf.__version__}')
    print(f'   Pandas: {pd.__version__}')
    print(f'   NumPy: {np.__version__}')
    sys.exit(0)
except Exception as e:
    print(f' Import failed: {e}')
    sys.exit(1)
"
        ;;
    web)
        echo " Starting Streamlit dashboard..."
        exec streamlit run src/webApp/app.py --server.address=0.0.0.0 --server.port=8501
        ;;
    shell)
        echo " Starting shell..."
        exec /bin/bash
        ;;
    *)
        echo "  Unknown mode: $MODE"
        echo "Available modes: pipeline, orchestrator, test, test-quick, web, shell"
        exit 1
        ;;
esac
EOF

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["pipeline"]
