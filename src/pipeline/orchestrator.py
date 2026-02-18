"""
Orquestador del Sistema de Predicción de Inflación.

Este módulo proporciona una interfaz de alto nivel (clase Orchestrator) para
ejecutar todas las operaciones del sistema de forma unificada.

Uso:
    python -m src.pipeline.orchestrator --mode full
    python -m src.pipeline.orchestrator --mode train --epochs 100
    python -m src.pipeline.orchestrator --mode status
"""

import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Importar todo desde el módulo core
from src.pipeline.core import (
    # Configuración
    CONFIG,
    VARIABLE_DESCRIPTIONS,
    ROOT_DIR,
    # Logging
    setup_logging,
    # Estado
    PipelineState,
    # Descarga
    download_all_data,
    # Modelo
    load_model,
    prepare_data,
    predict_future,
    finetune_model,
    train_model_from_scratch,
    forecast_all_covariates,
    get_scaler_stats,
    # Guardado
    save_predictions,
    plot_predictions,
    # Limpieza
    run_full_cleanup,
    cleanup_raw_data,
    cleanup_processed_data,
    cleanup_models,
    cleanup_predictions,
    # Pipeline
    run_pipeline,
)

from src.model.model import TFTModel


class Orchestrator:
    """
    Orquestador principal del sistema.
    
    Proporciona una interfaz unificada para todas las operaciones del sistema,
    incluyendo descarga, entrenamiento, predicción, fine-tuning, análisis y limpieza.
    
    Attributes:
        logger: Logger para registrar operaciones.
        state: Estado persistente del pipeline.
    
    Example:
        >>> orch = Orchestrator()
        >>> orch.run("status")
        >>> orch.run("full")
        >>> orch.run("train", epochs=100)
    """
    
    def __init__(self, logger: logging.Logger = None):
        """Inicializa el orquestador.
        
        Args:
            logger: Logger personalizado. Si es None, crea uno nuevo.
        """
        self.logger = logger or setup_logging()
        self.state = PipelineState()
    
    def _header(self, title: str):
        """Imprime un header formateado."""
        self.logger.info("=" * 70)
        self.logger.info(f"  {title}")
        self.logger.info("=" * 70)
    
    def mode_download(self) -> Dict[str, Any]:
        """Modo: Solo descarga de datos.
        
        Descarga datos de todas las fuentes (BanRep, FAO, FRED) y los consolida.
        
        Returns:
            Diccionario con resultado de la operación.
        """
        self._header("DESCARGA DE DATOS")
        
        success = download_all_data(self.logger)
        self.state.update("last_download", datetime.now().isoformat())
        
        # Limpiar datos antiguos
        cleanup_raw_data(self.logger)
        cleanup_processed_data(self.logger)
        
        status = "✓ Completada" if success else "⚠ Con errores"
        self.logger.info(f"\n{status}")
        
        return {"mode": "download", "success": success}
    
    def mode_train(self, epochs: int = None) -> Dict[str, Any]:
        """Modo: Entrenamiento de modelo desde cero.
        
        Entrena un nuevo modelo TFT con todos los datos disponibles.
        Esta es la única funcionalidad exclusiva del orchestrator.
        
        Args:
            epochs: Número de épocas de entrenamiento. Por defecto usa CONFIG.
        
        Returns:
            Diccionario con ruta del modelo y métricas.
        """
        self._header("ENTRENAMIENTO DE MODELO")
        
        model, model_path = train_model_from_scratch(self.logger, epochs)
        
        return {"mode": "train", "model_path": model_path, "epochs": epochs or CONFIG["train_epochs"]}
    
    def mode_predict(self, model_path: str = None) -> Dict[str, Any]:
        """Modo: Solo predicción.
        
        Genera predicciones para los próximos meses usando el modelo actual.
        
        Args:
            model_path: Ruta opcional al modelo. Si es None, usa el modelo actual.
        
        Returns:
            Diccionario con predicciones y rutas de archivos generados.
        """
        self._header("PREDICCIÓN")
        
        model, model_name = load_model(self.logger, self.state)
        df, feature_cols = prepare_data(self.logger)
        mean, std, mean_f, std_f = get_scaler_stats(df, feature_cols, self.logger)
        
        # Pronosticar todas las covariables futuras
        covariate_forecasts = forecast_all_covariates(df, logger=self.logger)
        
        pred_df = predict_future(
            model, df, feature_cols, mean, std, self.logger,
            covariate_forecasts=covariate_forecasts,
            mean_f=mean_f,
            std_f=std_f,
        )
        
        csv_path = save_predictions(pred_df, model_name, self.logger)
        plot_path = plot_predictions(df, pred_df, model_name, self.logger)
        
        self.state.update("last_prediction", datetime.now().isoformat())
        
        cleanup_predictions(self.logger)
        
        return {
            "mode": "predict",
            "predictions": pred_df.to_dict("records"),
            "csv_path": str(csv_path),
            "model": model_name,
        }
    
    def mode_finetune(self) -> Dict[str, Any]:
        """Modo: Fine-tuning del modelo.
        
        Realiza fine-tuning del modelo actual con los datos más recientes.
        
        Returns:
            Diccionario con nombre del modelo actualizado.
        """
        self._header("FINE-TUNING")
        
        model, model_name = load_model(self.logger, self.state)
        df, feature_cols = prepare_data(self.logger)
        
        model = finetune_model(model, df, feature_cols, self.logger, self.state)
        
        cleanup_models(self.logger)
        
        return {"mode": "finetune", "model": self.state.state["current_model"]}
    
    def mode_cleanup(self) -> Dict[str, Any]:
        """Modo: Limpiar archivos antiguos.
        
        Ejecuta el sistema de rotación eliminando versiones antiguas de datos,
        modelos y predicciones.
        
        Returns:
            Diccionario con resumen de archivos eliminados.
        """
        self._header("LIMPIEZA DE ARCHIVOS")
        
        summary = run_full_cleanup(self.logger)
        
        return {"mode": "cleanup", "summary": summary}
    
    def mode_analyze(self) -> Dict[str, Any]:
        """Modo: Análisis de features.
        
        Ejecuta el script de análisis de importancia de features.
        
        Returns:
            Diccionario con código de retorno del análisis.
        """
        self._header("ANÁLISIS DE FEATURES")
        
        result = subprocess.run(
            [sys.executable, str(ROOT_DIR / "feature_analysis.py")],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        
        self.logger.info(result.stdout)
        if result.stderr:
            self.logger.warning(result.stderr)
        
        return {"mode": "analyze", "returncode": result.returncode}
    
    def mode_full(self) -> Dict[str, Any]:
        """Modo: Pipeline completo.
        
        Ejecuta el pipeline completo: descarga, fine-tuning (si corresponde),
        predicción y limpieza.
        
        Returns:
            Diccionario con resultados del pipeline completo.
        """
        self._header("PIPELINE COMPLETO")
        self.logger.info(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = run_pipeline(
            download=True,
            predict=True,
            force_finetune=False,
            cleanup=True,
            logger=self.logger,
        )
        
        self._header("PIPELINE COMPLETADO")
        
        return {"mode": "full", "results": results}
    
    def mode_status(self) -> Dict[str, Any]:
        """Modo: Mostrar estado del sistema.
        
        Muestra información sobre el estado actual del sistema incluyendo
        últimas ejecuciones, modelo activo y datos disponibles.
        
        Returns:
            Diccionario con estado del sistema.
        """
        self._header("ESTADO DEL SISTEMA")
        
        state = self.state.state
        
        self.logger.info(f"Última descarga:    {state.get('last_download', 'Nunca')}")
        self.logger.info(f"Última predicción:  {state.get('last_prediction', 'Nunca')}")
        self.logger.info(f"Último fine-tuning: {state.get('last_finetune', 'Nunca')}")
        self.logger.info(f"Modelo activo:      {state.get('current_model', 'N/A')}")
        self.logger.info(f"Fine-tunings:       {state.get('finetune_count', 0)}")
        
        # Verificar datos disponibles
        try:
            df = TFTModel.load_latest_proc_csv(CONFIG["data_proc_dir"])
            self.logger.info(f"\nDatos disponibles:")
            self.logger.info(f"  Filas: {len(df)}")
            self.logger.info(f"  Rango: {df['date'].min()} → {df['date'].max()}")
            self.logger.info(f"  Features: {[c for c in df.columns if c != 'date']}")
        except Exception as e:
            self.logger.warning(f"No se pudieron cargar datos: {e}")
        
        return {"mode": "status", "state": state}
    
    def run(self, mode: str, **kwargs) -> Dict[str, Any]:
        """Ejecuta el modo especificado.
        
        Args:
            mode: Modo a ejecutar (download, train, predict, finetune, cleanup, analyze, full, status).
            **kwargs: Argumentos adicionales para el modo (ej: epochs para train).
        
        Returns:
            Diccionario con resultados del modo ejecutado.
        
        Raises:
            ValueError: Si el modo no es válido.
        """
        modes = {
            "download": self.mode_download,
            "train": lambda: self.mode_train(kwargs.get("epochs")),
            "predict": lambda: self.mode_predict(kwargs.get("model_path")),
            "finetune": self.mode_finetune,
            "cleanup": self.mode_cleanup,
            "analyze": self.mode_analyze,
            "full": self.mode_full,
            "status": self.mode_status,
        }
        
        if mode not in modes:
            raise ValueError(f"Modo desconocido: {mode}. Disponibles: {list(modes.keys())}")
        
        return modes[mode]()


def main():
    """Punto de entrada principal para CLI."""
    parser = argparse.ArgumentParser(
        description="Orquestador del Sistema de Predicción de Inflación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modos disponibles:
  full      Pipeline completo (descarga + finetune si toca + predicción)
  download  Solo descargar y consolidar datos
  train     Entrenar modelo desde cero
  predict   Solo realizar predicciones
  finetune  Forzar fine-tuning del modelo
  cleanup   Limpiar archivos antiguos
  analyze   Análisis de importancia de features
  status    Mostrar estado del sistema

Ejemplos:
  python -m src.pipeline.orchestrator --mode full
  python -m src.pipeline.orchestrator --mode train --epochs 100
  python -m src.pipeline.orchestrator --mode status
        """,
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="full",
        choices=["full", "download", "train", "predict", "finetune", "cleanup", "analyze", "status"],
        help="Modo de ejecución (default: full)",
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Número de épocas para entrenamiento",
    )
    
    parser.add_argument(
        "--model-path", "-p",
        type=str,
        default=None,
        help="Ruta al modelo para predicción",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Modo verbose",
    )
    
    args = parser.parse_args()
    
    # Ejecutar
    logger = setup_logging()
    orchestrator = Orchestrator(logger)
    
    try:
        result = orchestrator.run(
            mode=args.mode,
            epochs=args.epochs,
            model_path=args.model_path,
        )
        
        return 0 if result.get("success", True) else 1
        
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
