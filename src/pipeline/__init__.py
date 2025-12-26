"""
Pipeline - Módulo de orquestación del sistema de predicción de inflación.

Este módulo proporciona:
- core: Funciones y configuración central del pipeline
- orchestrator: Interfaz de alto nivel (clase Orchestrator)

Uso típico:
    from src.pipeline import Orchestrator
    orch = Orchestrator()
    orch.run("status")
    
    # O para acceso directo a funciones:
    from src.pipeline.core import run_pipeline, CONFIG
"""

# Core exports
from .core import (
    CONFIG,
    VARIABLE_DESCRIPTIONS,
    setup_logging,
    PipelineState,
    # Funciones de descarga
    download_all_data,
    # Funciones de modelo
    load_model,
    prepare_data,
    predict_future,
    finetune_model,
    train_model_from_scratch,
    # Funciones de guardado
    save_predictions,
    plot_predictions,
    # Funciones de limpieza
    cleanup_old_files,
    cleanup_raw_data,
    cleanup_processed_data,
    cleanup_models,
    cleanup_predictions,
    run_full_cleanup,
    # Pipeline principal
    run_pipeline,
)

# Orchestrator
from .orchestrator import Orchestrator

__all__ = [
    # Configuración
    "CONFIG",
    "VARIABLE_DESCRIPTIONS",
    # Logging
    "setup_logging",
    # Estado
    "PipelineState",
    # Funciones de descarga
    "download_all_data",
    # Funciones de modelo
    "load_model",
    "prepare_data",
    "predict_future",
    "finetune_model",
    "train_model_from_scratch",
    # Funciones de guardado
    "save_predictions",
    "plot_predictions",
    # Funciones de limpieza
    "cleanup_old_files",
    "cleanup_raw_data",
    "cleanup_processed_data",
    "cleanup_models",
    "cleanup_predictions",
    "run_full_cleanup",
    # Pipeline
    "run_pipeline",
    # Orchestrator
    "Orchestrator",
]
