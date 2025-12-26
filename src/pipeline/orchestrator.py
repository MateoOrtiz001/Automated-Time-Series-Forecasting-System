"""
Orquestador Principal - Sistema de Predicción de Inflación
============================================================

Este módulo coordina todas las operaciones del sistema:
- Extracción de datos (BanRep, FAO, Brent)
- Entrenamiento de modelos (TFT, LSTM, TST)
- Predicciones mensuales
- Fine-tuning periódico
- Análisis de features

Modos de ejecución:
    --mode full         Pipeline completo (descarga + predicción + finetune si toca)
    --mode download     Solo descargar y consolidar datos
    --mode train        Entrenar modelos desde cero
    --mode predict      Solo realizar predicciones
    --mode finetune     Forzar fine-tuning del modelo
    --mode analyze      Análisis de importancia de features
    --mode status       Mostrar estado del sistema

Uso:
    python src/pipeline/orchestrator.py --mode full
    python src/pipeline/orchestrator.py --mode predict --model tft
    python src/pipeline/orchestrator.py --mode train --epochs 200
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Configurar paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Imports del proyecto
from src.etl.dataExtractor import (
    extraer_suameca_sin_api,
    extraer_fao_indices,
    extraer_brent_fred,
    consolidar_suameca_json_a_csv_mensual,
)
from src.model.model import TFTModel


# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
CONFIG = {
    # Rutas principales
    "root_dir": ROOT_DIR,
    "data_raw_dir": ROOT_DIR / "data" / "raw",
    "data_proc_dir": ROOT_DIR / "data" / "proc",
    "models_dir": ROOT_DIR / "models",
    "results_dir": ROOT_DIR / "results",
    
    # Rutas del pipeline mensual
    "misc_dir": ROOT_DIR / "misc",
    "misc_models_dir": ROOT_DIR / "misc" / "models",
    "misc_results_dir": ROOT_DIR / "misc" / "results",
    "misc_logs_dir": ROOT_DIR / "misc" / "logs",
    
    # Configuración del modelo
    "target_col": "Inflacion_total",
    "lookback_steps": 12,
    "forecast_horizon": 1,
    "future_months": 12,
    
    # Entrenamiento
    "epochs": 200,
    "batch_size": 64,
    "patience": 40,
    "learning_rate": 1e-3,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    
    # Fine-tuning
    "finetune_interval_months": 3,
    "finetune_epochs": 50,
    "finetune_lr": 5e-4,
    
    # TFT específico
    "tft_units": 74,
    "tft_heads": 2,
    "tft_lstm_layers": 1,
    "tft_grn_layers": 2,
    "tft_dropout": 0.1,
}

# Descripciones de variables
VARIABLE_DESCRIPTIONS = {
    "Inflacion_total": "Inflación total (variación anual %)",
    "IPP": "Índice de Precios del Productor",
    "PIB_real_trimestral_2015_AE": "PIB real trimestral",
    "Tasa_interes_colocacion_total": "Tasa de interés de colocación",
    "TRM": "Tasa de cambio COP/USD",
    "Brent": "Precio del petróleo Brent (USD/barril)",
    "FAO": "Índice de precios de alimentos FAO",
}


# =============================================================================
# LOGGING
# =============================================================================
def setup_logging(log_dir: Path = None, level: int = logging.INFO) -> logging.Logger:
    """Configura logging a archivo y consola."""
    if log_dir is None:
        log_dir = CONFIG["misc_logs_dir"]
    
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"orchestrator_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Limpiar handlers existentes
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    return logging.getLogger(__name__)


# =============================================================================
# GESTIÓN DE ESTADO
# =============================================================================
class SystemState:
    """Gestiona el estado global del sistema."""
    
    def __init__(self, state_file: Path = None):
        self.state_file = state_file or (CONFIG["misc_dir"] / "system_state.json")
        self.state = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """Carga estado desde archivo."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "created_at": datetime.now().isoformat(),
            "last_download": None,
            "last_train": None,
            "last_predict": None,
            "last_finetune": None,
            "last_analyze": None,
            "active_model": "tft_base.keras",
            "model_versions": [],
            "total_predictions": 0,
            "total_downloads": 0,
        }
    
    def save(self):
        """Guarda estado a archivo."""
        self.state["updated_at"] = datetime.now().isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def update(self, key: str, value: Any):
        """Actualiza y guarda un valor."""
        self.state[key] = value
        self.save()
    
    def increment(self, key: str):
        """Incrementa un contador."""
        self.state[key] = self.state.get(key, 0) + 1
        self.save()


# =============================================================================
# MÓDULO DE DESCARGA DE DATOS
# =============================================================================
class DataDownloader:
    """Gestiona la descarga de datos de todas las fuentes."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.raw_dir = CONFIG["data_raw_dir"]
        self.proc_dir = CONFIG["data_proc_dir"]
    
    def download_banrep(self) -> bool:
        """Descarga datos de BanRep (SUAMECA)."""
        self.logger.info("Descargando datos de BanRep...")
        try:
            extraer_suameca_sin_api(
                output_dir=str(self.raw_dir / "banrep" / "suameca")
            )
            self.logger.info("✓ BanRep descargado")
            return True
        except Exception as e:
            self.logger.error(f"✗ Error BanRep: {e}")
            return False
    
    def download_fao(self) -> bool:
        """Descarga índice FAO de precios de alimentos."""
        self.logger.info("Descargando índice FAO...")
        try:
            external_dir = self.raw_dir / "external"
            external_dir.mkdir(parents=True, exist_ok=True)
            extraer_fao_indices(output_dir=str(external_dir))
            self.logger.info("✓ FAO descargado")
            return True
        except Exception as e:
            self.logger.error(f"✗ Error FAO: {e}")
            return False
    
    def download_brent(self) -> bool:
        """Descarga precio del Brent desde FRED."""
        self.logger.info("Descargando precio Brent...")
        try:
            external_dir = self.raw_dir / "external"
            external_dir.mkdir(parents=True, exist_ok=True)
            extraer_brent_fred(output_dir=str(external_dir))
            self.logger.info("✓ Brent descargado")
            return True
        except Exception as e:
            self.logger.error(f"✗ Error Brent: {e}")
            return False
    
    def consolidate(self) -> Dict[str, Any]:
        """Consolida todos los datos en CSV mensual."""
        self.logger.info("Consolidando datos...")
        try:
            result = consolidar_suameca_json_a_csv_mensual(
                input_dir=str(self.raw_dir / "banrep" / "suameca"),
                proc_dir=str(self.proc_dir),
                external_dir=str(self.raw_dir / "external"),
                verbose=False,
            )
            self.logger.info(f"✓ Datos consolidados: {result['global_start']} → {result['global_end']}")
            return result
        except Exception as e:
            self.logger.error(f"✗ Error consolidación: {e}")
            raise
    
    def download_all(self) -> Dict[str, bool]:
        """Ejecuta todas las descargas."""
        results = {
            "banrep": self.download_banrep(),
            "fao": self.download_fao(),
            "brent": self.download_brent(),
        }
        
        if any(results.values()):
            try:
                self.consolidate()
                results["consolidation"] = True
            except Exception:
                results["consolidation"] = False
        
        return results


# =============================================================================
# MÓDULO DE ENTRENAMIENTO
# =============================================================================
class ModelTrainer:
    """Gestiona el entrenamiento de modelos."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.proc_dir = CONFIG["data_proc_dir"]
        self.models_dir = CONFIG["models_dir"]
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> tuple:
        """Carga y prepara datos para entrenamiento."""
        df = TFTModel.load_latest_proc_csv(self.proc_dir)
        df = df.dropna().reset_index(drop=True)
        
        feature_cols = [c for c in df.columns if c != "date"]
        
        self.logger.info(f"Dataset: {len(df)} filas, {len(feature_cols)} features")
        self.logger.info(f"Rango: {df['date'].min()} → {df['date'].max()}")
        
        return df, feature_cols
    
    def prepare_datasets(self, df, feature_cols) -> tuple:
        """Prepara splits train/val/test."""
        X, y = TFTModel.make_supervised_windows(
            df=df,
            feature_cols=feature_cols,
            target_col=CONFIG["target_col"],
            lookback_steps=CONFIG["lookback_steps"],
            forecast_horizon=CONFIG["forecast_horizon"],
        )
        
        X_train, y_train, X_val, y_val, X_test, y_test = TFTModel.time_split(
            X, y,
            val_ratio=CONFIG["val_ratio"],
            test_ratio=CONFIG["test_ratio"],
        )
        
        X_train, X_val, X_test, mean, std = TFTModel.standardize_from_train(
            X_train, X_val, X_test
        )
        
        self.logger.info(f"Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}, Test: {len(X_test) if X_test is not None else 0}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, mean, std
    
    def train_tft(self, X_train, y_train, X_val, y_val, n_features: int, epochs: int = None) -> TFTModel:
        """Entrena modelo TFT."""
        self.logger.info("Entrenando modelo TFT...")
        
        tft = TFTModel(
            lookback_steps=CONFIG["lookback_steps"],
            forecast_horizon=CONFIG["forecast_horizon"],
            n_features=n_features,
            units=CONFIG["tft_units"],
            num_heads=CONFIG["tft_heads"],
            num_lstm_layers=CONFIG["tft_lstm_layers"],
            num_grn_layers=CONFIG["tft_grn_layers"],
            dropout_rate=CONFIG["tft_dropout"],
            num_quantiles=3,
        )
        
        tft.compile(learning_rate=CONFIG["learning_rate"])
        
        tft.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs or CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            patience=CONFIG["patience"],
            save_best=True,
            model_path=str(self.models_dir / "tft_best.keras"),
        )
        
        self.logger.info("✓ TFT entrenado y guardado")
        return tft
    
    def train_all(self, epochs: int = None) -> Dict[str, Any]:
        """Entrena todos los modelos."""
        df, feature_cols = self.load_data()
        X_train, y_train, X_val, y_val, X_test, y_test, mean, std = self.prepare_datasets(df, feature_cols)
        
        n_features = X_train.shape[-1]
        
        results = {}
        
        # TFT
        try:
            tft = self.train_tft(X_train, y_train, X_val, y_val, n_features, epochs)
            results["tft"] = {"status": "success", "path": str(self.models_dir / "tft_best.keras")}
        except Exception as e:
            self.logger.error(f"Error TFT: {e}")
            results["tft"] = {"status": "error", "error": str(e)}
        
        return results


# =============================================================================
# MÓDULO DE PREDICCIÓN
# =============================================================================
class Predictor:
    """Gestiona las predicciones."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.proc_dir = CONFIG["data_proc_dir"]
        self.results_dir = CONFIG["misc_results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_path: Path) -> TFTModel:
        """Carga un modelo TFT guardado."""
        import numpy as np
        
        df = TFTModel.load_latest_proc_csv(self.proc_dir)
        df = df.dropna().reset_index(drop=True)
        feature_cols = [c for c in df.columns if c != "date"]
        
        tft = TFTModel(
            lookback_steps=CONFIG["lookback_steps"],
            forecast_horizon=CONFIG["forecast_horizon"],
            n_features=len(feature_cols),
            units=CONFIG["tft_units"],
            num_heads=CONFIG["tft_heads"],
            num_lstm_layers=CONFIG["tft_lstm_layers"],
            num_grn_layers=CONFIG["tft_grn_layers"],
            dropout_rate=CONFIG["tft_dropout"],
            num_quantiles=3,
        )
        
        tft.build_model()
        tft.model.load_weights(str(model_path))
        
        return tft, df, feature_cols
    
    def predict_future(self, model, df, feature_cols, mean, std, n_months: int = 12):
        """Predice n_months hacia el futuro."""
        import numpy as np
        import pandas as pd
        
        data_features = df[feature_cols].to_numpy(dtype=np.float32)
        target_idx = feature_cols.index(CONFIG["target_col"])
        
        current_window = data_features[-CONFIG["lookback_steps"]:].copy()
        current_window_std = ((current_window - mean) / std).astype(np.float32)
        
        predictions = []
        last_date = df["date"].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=n_months,
            freq="MS"
        )
        
        for i in range(n_months):
            X_input = current_window_std[np.newaxis, :, :]
            
            pred_result = model.predict(X_input)
            pred_value = pred_result.get("median", pred_result.get("predictions"))
            pred_arr = np.ravel(pred_value)
            pred_value = float(pred_arr[1] if pred_arr.size >= 3 else pred_arr[0])
            
            lower = float(np.ravel(pred_result["lower"])[0]) if "lower" in pred_result else None
            upper = float(np.ravel(pred_result["upper"])[0]) if "upper" in pred_result else None
            
            predictions.append({
                "date": future_dates[i],
                "prediction": pred_value,
                "lower": lower,
                "upper": upper,
            })
            
            new_row = current_window[-1].copy()
            new_row[target_idx] = pred_value
            current_window = np.vstack([current_window[1:], new_row])
            current_window_std = ((current_window - mean) / std).astype(np.float32)
        
        return pd.DataFrame(predictions)
    
    def run_prediction(self, model_path: Path = None) -> Dict[str, Any]:
        """Ejecuta predicción completa."""
        import numpy as np
        import pandas as pd
        
        # Modelo por defecto
        if model_path is None:
            model_path = CONFIG["misc_models_dir"] / "tft_base.keras"
            if not model_path.exists():
                model_path = CONFIG["models_dir"] / "tft_best.keras"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró modelo en {model_path}")
        
        self.logger.info(f"Cargando modelo: {model_path.name}")
        model, df, feature_cols = self.load_model(model_path)
        
        # Calcular mean/std
        X, y = TFTModel.make_supervised_windows(
            df=df,
            feature_cols=feature_cols,
            target_col=CONFIG["target_col"],
            lookback_steps=CONFIG["lookback_steps"],
            forecast_horizon=CONFIG["forecast_horizon"],
        )
        X_train, _, _, _, _, _ = TFTModel.time_split(X, y)
        _, _, _, mean, std = TFTModel.standardize_from_train(X_train)
        
        # Predecir
        self.logger.info(f"Prediciendo {CONFIG['future_months']} meses...")
        pred_df = self.predict_future(model, df, feature_cols, mean, std, CONFIG["future_months"])
        
        # Guardar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.results_dir / f"predictions_{timestamp}.csv"
        pred_df.to_csv(csv_path, index=False)
        
        self.logger.info(f"✓ Predicciones guardadas: {csv_path.name}")
        
        # Log predicciones
        for _, row in pred_df.iterrows():
            ci = f" [{row['lower']:.2f}, {row['upper']:.2f}]" if pd.notna(row['lower']) else ""
            self.logger.info(f"  {row['date'].strftime('%Y-%m')}: {row['prediction']:.2f}%{ci}")
        
        return {
            "predictions": pred_df.to_dict("records"),
            "csv_path": str(csv_path),
            "model": model_path.name,
        }


# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================
class Orchestrator:
    """Orquestador principal del sistema."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or setup_logging()
        self.state = SystemState()
        self.downloader = DataDownloader(self.logger)
        self.trainer = ModelTrainer(self.logger)
        self.predictor = Predictor(self.logger)
    
    def _header(self, title: str):
        """Imprime header formateado."""
        self.logger.info("=" * 70)
        self.logger.info(f"  {title}")
        self.logger.info("=" * 70)
    
    def mode_download(self) -> Dict[str, Any]:
        """Modo: Solo descarga de datos."""
        self._header("DESCARGA DE DATOS")
        
        results = self.downloader.download_all()
        
        self.state.update("last_download", datetime.now().isoformat())
        self.state.increment("total_downloads")
        
        success = all(results.values())
        self.logger.info(f"\n{'✅' if success else '⚠️'} Descarga {'completada' if success else 'con errores'}")
        
        return {"mode": "download", "results": results, "success": success}
    
    def mode_train(self, epochs: int = None, model: str = "all") -> Dict[str, Any]:
        """Modo: Entrenamiento de modelos."""
        self._header("ENTRENAMIENTO DE MODELOS")
        
        results = self.trainer.train_all(epochs)
        
        self.state.update("last_train", datetime.now().isoformat())
        
        return {"mode": "train", "results": results}
    
    def mode_predict(self, model_path: str = None) -> Dict[str, Any]:
        """Modo: Predicción."""
        self._header("PREDICCIÓN")
        
        path = Path(model_path) if model_path else None
        results = self.predictor.run_prediction(path)
        
        self.state.update("last_predict", datetime.now().isoformat())
        self.state.increment("total_predictions")
        
        return {"mode": "predict", "results": results}
    
    def mode_finetune(self) -> Dict[str, Any]:
        """Modo: Fine-tuning del modelo."""
        self._header("FINE-TUNING")
        
        # Importar pipeline mensual para fine-tuning
        sys.path.insert(0, str(CONFIG["misc_dir"]))
        from monthly_pipeline import finetune_model, load_model, prepare_data, PipelineState
        
        pipeline_state = PipelineState()
        model, model_name = load_model(self.logger, pipeline_state)
        df, feature_cols, mean, std = prepare_data(self.logger)
        
        model = finetune_model(model, df, feature_cols, self.logger, pipeline_state)
        
        self.state.update("last_finetune", datetime.now().isoformat())
        self.state.update("active_model", pipeline_state.state["current_model"])
        
        return {"mode": "finetune", "model": pipeline_state.state["current_model"]}
    
    def mode_analyze(self) -> Dict[str, Any]:
        """Modo: Análisis de features."""
        self._header("ANÁLISIS DE FEATURES")
        
        # Ejecutar feature_analysis.py
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT_DIR / "feature_analysis.py")],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        
        self.logger.info(result.stdout)
        if result.stderr:
            self.logger.warning(result.stderr)
        
        self.state.update("last_analyze", datetime.now().isoformat())
        
        return {"mode": "analyze", "returncode": result.returncode}
    
    def mode_full(self) -> Dict[str, Any]:
        """Modo: Pipeline completo."""
        self._header("PIPELINE COMPLETO")
        self.logger.info(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        
        # 1. Descargar datos
        self.logger.info("\n[1/3] Descargando datos...")
        results["download"] = self.mode_download()
        
        # 2. Verificar si necesita fine-tuning
        pipeline_state = None
        try:
            sys.path.insert(0, str(CONFIG["misc_dir"]))
            from monthly_pipeline import PipelineState
            pipeline_state = PipelineState()
            needs_finetune = pipeline_state.should_finetune()
        except Exception:
            needs_finetune = False
        
        if needs_finetune:
            self.logger.info("\n[2/3] Fine-tuning requerido...")
            results["finetune"] = self.mode_finetune()
        else:
            self.logger.info("\n[2/3] Fine-tuning no requerido (último: hace menos de 3 meses)")
            results["finetune"] = {"skipped": True}
        
        # 3. Predecir
        self.logger.info("\n[3/3] Generando predicciones...")
        results["predict"] = self.mode_predict()
        
        self._header("PIPELINE COMPLETADO")
        
        return {"mode": "full", "results": results}
    
    def mode_status(self) -> Dict[str, Any]:
        """Modo: Mostrar estado del sistema."""
        self._header("ESTADO DEL SISTEMA")
        
        self.logger.info(f"Última descarga:    {self.state.state.get('last_download', 'Nunca')}")
        self.logger.info(f"Último entrenamiento: {self.state.state.get('last_train', 'Nunca')}")
        self.logger.info(f"Última predicción:  {self.state.state.get('last_predict', 'Nunca')}")
        self.logger.info(f"Último fine-tuning: {self.state.state.get('last_finetune', 'Nunca')}")
        self.logger.info(f"Modelo activo:      {self.state.state.get('active_model', 'N/A')}")
        self.logger.info(f"Total predicciones: {self.state.state.get('total_predictions', 0)}")
        self.logger.info(f"Total descargas:    {self.state.state.get('total_downloads', 0)}")
        
        # Verificar datos disponibles
        try:
            df = TFTModel.load_latest_proc_csv(CONFIG["data_proc_dir"])
            self.logger.info(f"\nDatos disponibles:")
            self.logger.info(f"  Filas: {len(df)}")
            self.logger.info(f"  Rango: {df['date'].min()} → {df['date'].max()}")
            self.logger.info(f"  Columnas: {list(df.columns)}")
        except Exception as e:
            self.logger.warning(f"No se pudieron cargar datos: {e}")
        
        return {"mode": "status", "state": self.state.state}
    
    def run(self, mode: str, **kwargs) -> Dict[str, Any]:
        """Ejecuta el modo especificado."""
        modes = {
            "download": self.mode_download,
            "train": lambda: self.mode_train(kwargs.get("epochs"), kwargs.get("model")),
            "predict": lambda: self.mode_predict(kwargs.get("model_path")),
            "finetune": self.mode_finetune,
            "analyze": self.mode_analyze,
            "full": self.mode_full,
            "status": self.mode_status,
        }
        
        if mode not in modes:
            raise ValueError(f"Modo desconocido: {mode}. Disponibles: {list(modes.keys())}")
        
        return modes[mode]()


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Orquestador del Sistema de Predicción de Inflación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modos disponibles:
  full      Pipeline completo (descarga + finetune si toca + predicción)
  download  Solo descargar y consolidar datos
  train     Entrenar modelos desde cero
  predict   Solo realizar predicciones
  finetune  Forzar fine-tuning del modelo
  analyze   Análisis de importancia de features
  status    Mostrar estado del sistema

Ejemplos:
  python orchestrator.py --mode full
  python orchestrator.py --mode train --epochs 100
  python orchestrator.py --mode predict --model-path misc/models/tft_base.keras
        """,
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="full",
        choices=["full", "download", "train", "predict", "finetune", "analyze", "status"],
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
        "--model", "-M",
        type=str,
        default="tft",
        choices=["tft"],
        help="Modelo a entrenar (default: tft)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Modo verbose",
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)
    
    # Ejecutar
    orchestrator = Orchestrator(logger)
    
    try:
        result = orchestrator.run(
            mode=args.mode,
            epochs=args.epochs,
            model_path=args.model_path,
            model=args.model,
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
