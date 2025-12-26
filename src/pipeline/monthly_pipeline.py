#!/usr/bin/env python
"""
Pipeline mensual para predicción de inflación.

Este script es un wrapper simple que invoca las funciones del módulo core.
Mantiene la interfaz CLI original para compatibilidad.

Uso:
    python -m src.pipeline.monthly_pipeline                    # Ejecución completa
    python -m src.pipeline.monthly_pipeline --download-only    # Solo descargar datos
    python -m src.pipeline.monthly_pipeline --predict-only     # Solo predicción
    python -m src.pipeline.monthly_pipeline --finetune         # Forzar fine-tuning
    python -m src.pipeline.monthly_pipeline --cleanup-only     # Solo limpiar archivos antiguos
    python -m src.pipeline.monthly_pipeline --no-cleanup       # Ejecutar sin limpieza automática
"""

import sys
import json
import argparse
from pathlib import Path

# Configurar paths para imports
SCRIPT_DIR = Path(__file__).resolve().parent  # src/pipeline
SRC_DIR = SCRIPT_DIR.parent                    # src/
ROOT_DIR = SRC_DIR.parent                      # raíz del proyecto
sys.path.insert(0, str(ROOT_DIR))

# Importar desde el módulo core
from src.pipeline.core import (
    CONFIG,
    setup_logging,
    run_pipeline,
    run_full_cleanup,
)


def main():
    """Punto de entrada principal para el pipeline mensual."""
    parser = argparse.ArgumentParser(
        description="Pipeline mensual para predicción de inflación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python -m src.pipeline.monthly_pipeline                    # Ejecución completa
  python -m src.pipeline.monthly_pipeline --download-only    # Solo descargar datos
  python -m src.pipeline.monthly_pipeline --predict-only     # Solo predicción
  python -m src.pipeline.monthly_pipeline --finetune         # Forzar fine-tuning
  python -m src.pipeline.monthly_pipeline --cleanup-only     # Solo limpiar archivos antiguos
  python -m src.pipeline.monthly_pipeline --no-cleanup       # Ejecutar sin limpieza automática

Sistema de rotación:
  El pipeline mantiene automáticamente solo las últimas 2 versiones de:
  - Archivos de datos procesados (data/proc/*.csv)
  - Modelos fine-tuned (misc/models/tft_finetuned_*.keras)
  - Predicciones (misc/results/predictions_*.csv)
  - Archivos raw por serie (data/raw/banrep/suameca/*.json)
        """,
    )
    
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Solo descargar y consolidar datos",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Solo realizar predicción (sin descargar)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Forzar fine-tuning del modelo",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Omitir descarga de datos",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Solo ejecutar limpieza de archivos antiguos (sin descargar ni predecir)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="No ejecutar limpieza automática de archivos antiguos",
    )
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Modo solo limpieza
    if args.cleanup_only:
        logger.info("\n" + "=" * 70)
        logger.info("MODO: SOLO LIMPIEZA")
        logger.info("=" * 70)
        summary = run_full_cleanup(logger)
        logger.info(f"\nResumen de limpieza: {json.dumps(summary, indent=2, default=str)}")
        return 0
    
    # Determinar qué ejecutar
    download = not args.predict_only and not args.no_download
    predict = not args.download_only
    force_finetune = args.finetune
    cleanup = not args.no_cleanup
    
    results = run_pipeline(
        download=download,
        predict=predict,
        force_finetune=force_finetune,
        cleanup=cleanup,
        logger=logger,
    )
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
