"""
Core del Pipeline de Predicción de Inflación.

Este módulo contiene toda la lógica central del sistema:
- Configuración global
- Funciones de descarga de datos
- Funciones de modelo (carga, predicción, fine-tuning)
- Sistema de limpieza/rotación de archivos
- Estado persistente del pipeline
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para servidores
import matplotlib.pyplot as plt

# Configurar paths
PIPELINE_DIR = Path(__file__).resolve().parent  # src/pipeline
SRC_DIR = PIPELINE_DIR.parent                    # src/
ROOT_DIR = SRC_DIR.parent                        # raíz del proyecto
MISC_DIR = ROOT_DIR / "misc"                     # misc/ (para logs, models, results)

# Imports del proyecto
from src.etl.dataExtractor import (
    extraer_suameca_sin_api,
    extraer_fao_indices,
    extraer_brent_fred,
    consolidar_suameca_json_a_csv_mensual,
)
from src.model.model import TFTModel


# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================
CONFIG = {
    # Rutas principales
    "root_dir": ROOT_DIR,
    "src_dir": SRC_DIR,
    "pipeline_dir": PIPELINE_DIR,
    "misc_dir": MISC_DIR,
    
    # Rutas de datos
    "data_raw_dir": ROOT_DIR / "data" / "raw",
    "data_proc_dir": ROOT_DIR / "data" / "proc",
    
    # Rutas del pipeline (fine-tuned models, predictions, logs)
    "pipeline_models_dir": MISC_DIR / "models",
    "pipeline_results_dir": MISC_DIR / "results",
    "pipeline_logs_dir": MISC_DIR / "logs",
    "pipeline_state_file": PIPELINE_DIR / "pipeline_state.json",
    
    # Rutas en raíz (para modelos base y resultados de análisis)
    "models_dir": ROOT_DIR / "models",
    "results_dir": ROOT_DIR / "results",
    
    # Modelo
    "base_model": "tft_base.keras",
    "lookback_steps": 12,
    "forecast_horizon": 1,
    "target_col": "Inflacion_total",
    "future_months": 12,
    
    # Fine-tuning
    "finetune_interval_months": 6,
    "finetune_epochs": 50,
    "finetune_patience": 15,
    "finetune_batch_size": 32,
    "finetune_lr": 5e-4,
    "finetune_window_months": 120,  # Ventana móvil: últimos 10 años
    "finetune_val_ratio": 0.1,      # Val del inicio (no del final)
    "finetune_test_ratio": 0.0,     # Sin test en fine-tuning
    
    # Entrenamiento desde cero
    "train_epochs": 200,
    "train_batch_size": 64,
    "train_patience": 40,
    "train_lr": 1e-3,
    
    # Split ratios
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    
    # Sistema de rotación (mantener solo N versiones de cada tipo de archivo)
    "max_data_versions": 2,
    "max_model_versions": 2,
    "max_prediction_versions": 2,
    "max_raw_versions": 2,
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


# ============================================================================
# LOGGING
# ============================================================================
def setup_logging(log_dir: Path = None) -> logging.Logger:
    """Configura logging a archivo y consola.
    
    Args:
        log_dir: Directorio para logs. Por defecto usa misc/logs.
    
    Returns:
        Logger configurado.
    """
    log_dir = log_dir or CONFIG["pipeline_logs_dir"]
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m')}.log"
    
    # Crear directorio si no existe
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    logger = logging.getLogger(__name__)
    return logger


# ============================================================================
# ESTADO DEL PIPELINE
# ============================================================================
class PipelineState:
    """Maneja el estado persistente del pipeline."""
    
    def __init__(self, state_file: Path = None):
        self.state_file = state_file or CONFIG["pipeline_state_file"]
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Carga estado desde archivo o crea uno nuevo."""
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "last_download": None,
            "last_prediction": None,
            "last_finetune": None,
            "finetune_count": 0,
            "current_model": CONFIG["base_model"],
            "execution_history": [],
        }
    
    def save(self):
        """Guarda estado a archivo."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def update(self, key: str, value: Any):
        """Actualiza un valor del estado."""
        self.state[key] = value
        self.save()
    
    def should_finetune(self) -> bool:
        """Determina si es momento de hacer fine-tuning."""
        if self.state["last_finetune"] is None:
            return True
        
        last_finetune = datetime.fromisoformat(self.state["last_finetune"])
        months_since = (datetime.now() - last_finetune).days / 30
        
        return months_since >= CONFIG["finetune_interval_months"]
    
    def add_execution(self, execution_type: str, details: Dict):
        """Añade una ejecución al historial."""
        self.state["execution_history"].append({
            "timestamp": datetime.now().isoformat(),
            "type": execution_type,
            **details,
        })
        # Mantener solo las últimas 100 ejecuciones
        self.state["execution_history"] = self.state["execution_history"][-100:]
        self.save()


# ============================================================================
# LIMPIEZA DE ARCHIVOS (SISTEMA DE ROTACIÓN)
# ============================================================================
def cleanup_old_files(
    directory: Path,
    pattern: str,
    max_versions: int,
    logger: logging.Logger,
    sort_by_date_in_name: bool = True,
) -> List[Path]:
    """Elimina archivos antiguos manteniendo solo las últimas `max_versions` versiones.
    
    Args:
        directory: Directorio donde buscar archivos.
        pattern: Patrón glob para filtrar archivos (ej: "*.csv", "tft_finetuned_*.keras").
        max_versions: Número máximo de versiones a mantener.
        logger: Logger para registrar operaciones.
        sort_by_date_in_name: Si True, ordena por timestamp en el nombre del archivo.
                              Si False, ordena por fecha de modificación del archivo.
    
    Returns:
        Lista de archivos eliminados.
    """
    if not directory.exists():
        return []
    
    files = list(directory.glob(pattern))
    
    if len(files) <= max_versions:
        return []
    
    if sort_by_date_in_name:
        def extract_timestamp(p: Path) -> str:
            name = p.stem
            parts = name.split("__")
            for part in reversed(parts):
                if len(part) >= 8 and part[:8].isdigit():
                    return part
            parts = name.split("_")
            for part in parts:
                if len(part) == 6 and part.isdigit():
                    return part
            return name
        
        files = sorted(files, key=extract_timestamp, reverse=True)
    else:
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    
    files_to_delete = files[max_versions:]
    deleted = []
    
    for f in files_to_delete:
        try:
            f.unlink()
            deleted.append(f)
            logger.info(f"   [CLEANUP] Eliminado: {f.name}")
        except Exception as e:
            logger.warning(f"   [CLEANUP] No se pudo eliminar {f.name}: {e}")
    
    return deleted


def cleanup_raw_data(logger: logging.Logger) -> Dict[str, List[Path]]:
    """Limpia archivos raw antiguos manteniendo solo las últimas N versiones por serie."""
    result = {}
    
    # Limpiar archivos JSON de BanRep/SUAMECA
    suameca_dir = CONFIG["data_raw_dir"] / "banrep" / "suameca"
    if suameca_dir.exists():
        series_files: Dict[str, List[Path]] = defaultdict(list)
        
        for f in suameca_dir.glob("*.json"):
            if "manifest" in f.name.lower():
                continue
            parts = f.stem.split("__")
            if len(parts) >= 3:
                serie = parts[2]
                series_files[serie].append(f)
        
        for serie, files in series_files.items():
            if len(files) <= CONFIG["max_raw_versions"]:
                continue
            
            files = sorted(files, key=lambda p: p.stem.split("__")[-1] if "__" in p.stem else "", reverse=True)
            to_delete = files[CONFIG["max_raw_versions"]:]
            
            deleted = []
            for f in to_delete:
                try:
                    f.unlink()
                    deleted.append(f)
                    logger.info(f"   [CLEANUP RAW] Eliminado: {f.name}")
                except Exception as e:
                    logger.warning(f"   [CLEANUP RAW] No se pudo eliminar {f.name}: {e}")
            
            if deleted:
                result[serie] = deleted
    
    # Limpiar manifests antiguos
    manifests = sorted(
        suameca_dir.glob("*manifest*.json"),
        key=lambda p: p.stem.split("__")[-1] if "__" in p.stem else "",
        reverse=True
    )
    if len(manifests) > CONFIG["max_raw_versions"]:
        for f in manifests[CONFIG["max_raw_versions"]:]:
            try:
                f.unlink()
                logger.info(f"   [CLEANUP RAW] Eliminado manifest: {f.name}")
                result.setdefault("manifests", []).append(f)
            except Exception as e:
                logger.warning(f"   [CLEANUP RAW] No se pudo eliminar {f.name}: {e}")
    
    # Limpiar archivos externos (Brent, FAO)
    external_dir = CONFIG["data_raw_dir"] / "external"
    if external_dir.exists():
        for prefix in ["brent__", "fao__"]:
            files = sorted(external_dir.glob(f"{prefix}*.csv"), reverse=True)
            if len(files) > CONFIG["max_raw_versions"]:
                for f in files[CONFIG["max_raw_versions"]:]:
                    try:
                        f.unlink()
                        logger.info(f"   [CLEANUP RAW] Eliminado: {f.name}")
                        result.setdefault("external", []).append(f)
                    except Exception as e:
                        logger.warning(f"   [CLEANUP RAW] No se pudo eliminar {f.name}: {e}")
    
    return result


def cleanup_processed_data(logger: logging.Logger) -> List[Path]:
    """Limpia archivos CSV procesados antiguos."""
    return cleanup_old_files(
        directory=CONFIG["data_proc_dir"],
        pattern="*.csv",
        max_versions=CONFIG["max_data_versions"],
        logger=logger,
    )


def cleanup_models(logger: logging.Logger) -> List[Path]:
    """Limpia modelos fine-tuned antiguos (NO elimina tft_base.keras)."""
    return cleanup_old_files(
        directory=CONFIG["pipeline_models_dir"],
        pattern="tft_finetuned_*.keras",
        max_versions=CONFIG["max_model_versions"],
        logger=logger,
    )


def cleanup_predictions(logger: logging.Logger) -> List[Path]:
    """Limpia archivos de predicciones antiguos (excluyendo *_latest.csv)."""
    deleted = []
    
    results_dir = CONFIG["pipeline_results_dir"]
    if not results_dir.exists():
        return deleted
    
    # CSVs de predicciones (excluir predictions_latest.csv)
    csv_files = [f for f in results_dir.glob("predictions_*.csv") 
                 if not f.name.endswith("_latest.csv") and f.name != "predictions_latest.csv"]
    
    if len(csv_files) > CONFIG["max_prediction_versions"]:
        # Ordenar por timestamp en nombre (formato: predictions_YYYYMMDD_HHMMSS.csv)
        csv_files = sorted(csv_files, key=lambda p: p.stem.split("_", 1)[1] if "_" in p.stem else "", reverse=True)
        for f in csv_files[CONFIG["max_prediction_versions"]:]:
            try:
                f.unlink()
                deleted.append(f)
                logger.info(f"   [CLEANUP] Eliminado: {f.name}")
            except Exception as e:
                logger.warning(f"   [CLEANUP] No se pudo eliminar {f.name}: {e}")
    
    # Gráficos de predicciones
    png_files = [f for f in results_dir.glob("predictions_plot_*.png")]
    
    if len(png_files) > CONFIG["max_prediction_versions"]:
        png_files = sorted(png_files, key=lambda p: p.stem.split("_", 2)[2] if "_" in p.stem else "", reverse=True)
        for f in png_files[CONFIG["max_prediction_versions"]:]:
            try:
                f.unlink()
                deleted.append(f)
                logger.info(f"   [CLEANUP] Eliminado: {f.name}")
            except Exception as e:
                logger.warning(f"   [CLEANUP] No se pudo eliminar {f.name}: {e}")
    
    return deleted


def run_full_cleanup(logger: logging.Logger) -> Dict[str, Any]:
    """Ejecuta limpieza completa de archivos antiguos.
    
    Returns:
        Resumen de archivos eliminados.
    """
    logger.info("\n[CLEANUP] Limpiando archivos antiguos...")
    
    summary = {
        "raw_data": {},
        "processed_data": [],
        "models": [],
        "predictions": [],
    }
    
    try:
        summary["raw_data"] = cleanup_raw_data(logger)
    except Exception as e:
        logger.warning(f"   [CLEANUP] Error limpiando datos raw: {e}")
    
    try:
        summary["processed_data"] = [str(p) for p in cleanup_processed_data(logger)]
    except Exception as e:
        logger.warning(f"   [CLEANUP] Error limpiando datos procesados: {e}")
    
    try:
        summary["models"] = [str(p) for p in cleanup_models(logger)]
    except Exception as e:
        logger.warning(f"   [CLEANUP] Error limpiando modelos: {e}")
    
    try:
        summary["predictions"] = [str(p) for p in cleanup_predictions(logger)]
    except Exception as e:
        logger.warning(f"   [CLEANUP] Error limpiando predicciones: {e}")
    
    total = (
        sum(len(v) for v in summary["raw_data"].values()) +
        len(summary["processed_data"]) +
        len(summary["models"]) +
        len(summary["predictions"])
    )
    
    logger.info(f"[CLEANUP] Total archivos eliminados: {total}")
    
    return summary


# ============================================================================
# DESCARGA DE DATOS
# ============================================================================
def download_all_data(logger: logging.Logger) -> bool:
    """Descarga todos los datos necesarios."""
    logger.info("=" * 60)
    logger.info("INICIANDO DESCARGA DE DATOS")
    logger.info("=" * 60)
    
    success = True
    
    # Datos BanRep (SUAMECA)
    logger.info("\n[1/3] Descargando datos de BanRep (SUAMECA)...")
    try:
        extraer_suameca_sin_api(output_dir=str(CONFIG["data_raw_dir"] / "banrep" / "suameca"))
        logger.info("✓ BanRep descargado correctamente")
    except Exception as e:
        logger.error(f"✗ Error descargando BanRep: {e}")
        success = False
    
    # Datos FAO
    logger.info("\n[2/3] Descargando índice FAO...")
    try:
        external_dir = CONFIG["data_raw_dir"] / "external"
        external_dir.mkdir(parents=True, exist_ok=True)
        extraer_fao_indices(output_dir=str(external_dir))
        logger.info("✓ FAO descargado correctamente")
    except Exception as e:
        logger.error(f"✗ Error descargando FAO: {e}")
        success = False
    
    # Datos Brent (FRED)
    logger.info("\n[3/3] Descargando precio Brent (FRED)...")
    try:
        external_dir = CONFIG["data_raw_dir"] / "external"
        extraer_brent_fred(output_dir=str(external_dir))
        logger.info("✓ Brent descargado correctamente")
    except Exception as e:
        logger.error(f"✗ Error descargando Brent: {e}")
        success = False
    
    # Consolidar datos
    if success:
        logger.info("\n[4/4] Consolidando datos procesados...")
        try:
            result = consolidar_suameca_json_a_csv_mensual(
                input_dir=str(CONFIG["data_raw_dir"] / "banrep" / "suameca"),
                proc_dir=str(CONFIG["data_proc_dir"]),
                external_dir=str(CONFIG["data_raw_dir"] / "external"),
            )
            logger.info("✓ Datos consolidados correctamente")
            
            # Copiar a latest.csv (para Streamlit)
            output_csv = Path(result["output_csv_path"])
            latest_csv = CONFIG["data_proc_dir"] / "latest.csv"
            shutil.copy(output_csv, latest_csv)
            logger.info(f"✓ Actualizado: {latest_csv.name}")
        except Exception as e:
            logger.error(f"✗ Error consolidando datos: {e}")
            success = False
    
    return success


# ============================================================================
# FUNCIONES DE MODELO
# ============================================================================
def load_model(logger: logging.Logger, state: PipelineState) -> Tuple[TFTModel, str]:
    """Carga el modelo más reciente (base o fine-tuned)."""
    models_dir = CONFIG["pipeline_models_dir"]
    
    current_model = state.state["current_model"]
    model_path = models_dir / current_model
    
    if not model_path.exists():
        model_path = models_dir / CONFIG["base_model"]
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo base en {model_path}")
    
    logger.info(f"Cargando modelo: {model_path.name}")
    
    # Cargar datos para obtener n_features
    df = TFTModel.load_latest_proc_csv(CONFIG["data_proc_dir"])
    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != "date"]
    n_features = len(feature_cols)
    
    # Crear instancia del modelo
    tft = TFTModel(
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
        n_features=n_features,
        units=74,
        num_heads=2,
        num_lstm_layers=1,
        num_grn_layers=2,
        dropout_rate=0.1,
        num_quantiles=3,
    )
    
    # Construir y cargar pesos
    tft.build_model()
    tft.model.load_weights(str(model_path))
    
    return tft, model_path.name


def prepare_data(logger: logging.Logger) -> Tuple[pd.DataFrame, List, np.ndarray, np.ndarray]:
    """Prepara los datos para predicción/entrenamiento."""
    df = TFTModel.load_latest_proc_csv(CONFIG["data_proc_dir"])
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = [c for c in df.columns if c != "date"]
    
    logger.info(f"Dataset cargado: {len(df)} filas")
    logger.info(f"Rango: {df['date'].min()} → {df['date'].max()}")
    logger.info(f"Features ({len(feature_cols)}):")
    for col in feature_cols:
        desc = VARIABLE_DESCRIPTIONS.get(col, col)
        logger.info(f"   • {col}: {desc}")
    
    # Calcular media y std para estandarización
    X, y = TFTModel.make_supervised_windows(
        df=df,
        feature_cols=feature_cols,
        target_col=CONFIG["target_col"],
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
    )
    
    X_train, _, X_val, _, X_test, _ = TFTModel.time_split(
        X, y, val_ratio=CONFIG["val_ratio"], test_ratio=CONFIG["test_ratio"]
    )
    
    _, _, _, mean, std = TFTModel.standardize_from_train(X_train, X_val, X_test)
    
    return df, feature_cols, mean, std


def predict_future(
    model: TFTModel,
    df: pd.DataFrame,
    feature_cols: List,
    mean: np.ndarray,
    std: np.ndarray,
    logger: logging.Logger,
    n_months: int = 12,
) -> pd.DataFrame:
    """Realiza predicciones iterativas para los próximos n_months meses."""
    logger.info(f"\nPrediciendo {n_months} meses futuros...")
    
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
    
    pred_df = pd.DataFrame(predictions)
    
    logger.info(f"\nPredicciones ({CONFIG['target_col']}):")
    for _, row in pred_df.iterrows():
        ci = f" [{row['lower']:.2f}, {row['upper']:.2f}]" if row['lower'] else ""
        logger.info(f"  {row['date'].strftime('%Y-%m')}: {row['prediction']:.2f}%{ci}")
    
    return pred_df


def finetune_model(
    model: TFTModel,
    df: pd.DataFrame,
    feature_cols: List,
    logger: logging.Logger,
    state: PipelineState,
) -> TFTModel:
    """
    Realiza fine-tuning del modelo con los datos más recientes.
    
    Estrategia:
    - Ventana móvil: usa solo los últimos N meses (configurable).
    - Split: val al INICIO de la ventana (para early stopping), train con el resto.
    - Sin test: los datos más recientes SÍ participan en el entrenamiento.
    
    Para evaluación del modelo fine-tuneado, usar backtest walk-forward por separado.
    """
    logger.info("\n" + "=" * 60)
    logger.info("INICIANDO FINE-TUNING DEL MODELO")
    logger.info("=" * 60)
    
    # --- Ventana móvil: recortar a los últimos N meses ---
    window_months = CONFIG.get("finetune_window_months", 120)  # default 10 años
    original_len = len(df)
    
    if len(df) > window_months:
        df = df.iloc[-window_months:].reset_index(drop=True)
        logger.info(f"Ventana móvil: usando últimos {window_months} meses ({len(df)} filas de {original_len} originales)")
    else:
        logger.info(f"Usando todos los datos disponibles ({len(df)} meses, menor que ventana de {window_months})")
    
    X, y = TFTModel.make_supervised_windows(
        df=df,
        feature_cols=feature_cols,
        target_col=CONFIG["target_col"],
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
    )
    
    # --- Split para fine-tuning: val al inicio, sin test ---
    val_ratio = CONFIG.get("finetune_val_ratio", 0.1)
    X_train, y_train, X_val, y_val = TFTModel.time_split_finetune(
        X, y, val_ratio=val_ratio
    )
    
    # Estandarizar (nota: val está al inicio pero estandarizamos con train para consistencia)
    X_train, X_val, _, mean, std = TFTModel.standardize_from_train(X_train, X_val, None)
    
    logger.info(f"Train: {X_train.shape[0]} samples (datos más recientes incluidos)")
    logger.info(f"Val: {X_val.shape[0]} samples (del inicio de la ventana, para early stopping)")
    logger.info(f"Test: 0 samples (sin test en fine-tuning; usar backtest para evaluación)")
    
    timestamp = datetime.now().strftime("%Y%m")
    finetune_count = state.state["finetune_count"] + 1
    new_model_name = f"tft_finetuned_{timestamp}_v{finetune_count}.keras"
    new_model_path = CONFIG["pipeline_models_dir"] / new_model_name
    
    CONFIG["pipeline_models_dir"].mkdir(parents=True, exist_ok=True)
    
    model.compile(learning_rate=CONFIG["finetune_lr"])
    
    logger.info(f"\nEntrenando {CONFIG['finetune_epochs']} épocas con lr={CONFIG['finetune_lr']}...")
    
    history = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=CONFIG["finetune_epochs"],
        batch_size=CONFIG["finetune_batch_size"],
        patience=CONFIG["finetune_patience"],
        save_best=True,
        model_path=str(new_model_path),
    )
    
    logger.info("\nEvaluación post fine-tuning (in-sample):")
    logger.info("  Nota: Sin test set. Para evaluación out-of-sample usar backtest walk-forward.")
    
    for name, X, y_true in [("Train", X_train, y_train), ("Val", X_val, y_val)]:
        if X is None or len(X) == 0:
            continue
        preds = model.predict(X)
        y_pred = preds.get("median", preds.get("predictions"))
        y_pred = np.ravel(y_pred)
        if y_pred.size >= 3 * len(y_true):
            y_pred = y_pred.reshape(-1, 3)[:, 1]
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        logger.info(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    state.update("last_finetune", datetime.now().isoformat())
    state.update("finetune_count", finetune_count)
    state.update("current_model", new_model_name)
    
    logger.info(f"\n✓ Modelo guardado: {new_model_name}")
    
    return model


def train_model_from_scratch(
    logger: logging.Logger,
    epochs: int = None,
) -> Tuple[TFTModel, str]:
    """Entrena un modelo TFT desde cero."""
    logger.info("\n" + "=" * 60)
    logger.info("ENTRENAMIENTO DE MODELO DESDE CERO")
    logger.info("=" * 60)
    
    epochs = epochs or CONFIG["train_epochs"]
    models_dir = CONFIG["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar y preparar datos
    df, feature_cols, mean, std = prepare_data(logger)
    
    X, y = TFTModel.make_supervised_windows(
        df=df,
        feature_cols=feature_cols,
        target_col=CONFIG["target_col"],
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = TFTModel.time_split(
        X, y, val_ratio=CONFIG["val_ratio"], test_ratio=CONFIG["test_ratio"]
    )
    
    X_train, X_val, X_test, _, _ = TFTModel.standardize_from_train(X_train, X_val, X_test)
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}, Test: {len(X_test) if X_test is not None else 0}")
    
    # Crear y entrenar modelo
    tft = TFTModel(
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
        n_features=len(feature_cols),
        units=74,
        num_heads=2,
        num_lstm_layers=1,
        num_grn_layers=2,
        dropout_rate=0.1,
        num_quantiles=3,
    )
    
    tft.compile(learning_rate=CONFIG["train_lr"])
    
    model_path = models_dir / "tft_best.keras"
    tft.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=CONFIG["train_batch_size"],
        patience=CONFIG["train_patience"],
        save_best=True,
        model_path=str(model_path),
    )
    
    logger.info(f"✓ Modelo guardado: {model_path}")
    
    return tft, str(model_path)


# ============================================================================
# GUARDADO DE RESULTADOS
# ============================================================================
def save_predictions(pred_df: pd.DataFrame, model_name: str, logger: logging.Logger) -> Path:
    """Guarda las predicciones en CSV (con timestamp y copia a latest)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pred_df = pred_df.copy()
    pred_df["model"] = model_name
    pred_df["generated_at"] = timestamp
    
    CONFIG["pipeline_results_dir"].mkdir(parents=True, exist_ok=True)
    
    # Guardar con timestamp
    output_path = CONFIG["pipeline_results_dir"] / f"predictions_{timestamp}.csv"
    pred_df.to_csv(output_path, index=False)
    
    # También guardar como latest (para Streamlit)
    latest_path = CONFIG["pipeline_results_dir"] / "predictions_latest.csv"
    pred_df.to_csv(latest_path, index=False)
    
    logger.info(f"✓ Predicciones guardadas: {output_path.name}")
    
    return output_path


def plot_predictions(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    model_name: str,
    logger: logging.Logger,
) -> Path:
    """Genera gráfico de predicciones."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    recent_df = df.tail(36)
    ax.plot(
        recent_df["date"],
        recent_df[CONFIG["target_col"]],
        label="Histórico",
        color="blue",
        linewidth=2,
    )
    
    ax.plot(
        pred_df["date"],
        pred_df["prediction"],
        label=f"Predicción ({model_name})",
        color="red",
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=4,
    )
    
    if pred_df["lower"].notna().any():
        ax.fill_between(
            pred_df["date"],
            pred_df["lower"],
            pred_df["upper"],
            alpha=0.2,
            color="red",
            label="IC 80%",
        )
    
    ax.axvline(x=df["date"].iloc[-1], color="gray", linestyle=":", alpha=0.7)
    
    ax.set_title(f"Predicción de {CONFIG['target_col']} - {datetime.now().strftime('%Y-%m-%d')}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(f"{CONFIG['target_col']} (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = CONFIG["pipeline_results_dir"] / f"predictions_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"✓ Gráfico guardado: {plot_path.name}")
    
    return plot_path


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================
def run_pipeline(
    download: bool = True,
    predict: bool = True,
    force_finetune: bool = False,
    cleanup: bool = True,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """Ejecuta el pipeline completo.
    
    Args:
        download: Si descarga nuevos datos.
        predict: Si genera predicciones.
        force_finetune: Si fuerza fine-tuning del modelo.
        cleanup: Si ejecuta limpieza de archivos antiguos después de cada operación.
        logger: Logger para registrar operaciones.
    
    Returns:
        Diccionario con resultados del pipeline.
    """
    if logger is None:
        logger = setup_logging()
    
    state = PipelineState()
    results = {"success": True, "errors": [], "cleanup": {}}
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE MENSUAL DE PREDICCIÓN DE INFLACIÓN")
    logger.info("=" * 70)
    logger.info(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Modelo actual: {state.state['current_model']}")
    logger.info(f"Último fine-tuning: {state.state['last_finetune'] or 'Nunca'}")
    logger.info(f"Sistema de rotación: max_data={CONFIG['max_data_versions']}, max_models={CONFIG['max_model_versions']}")
    
    # Descarga de datos
    if download:
        try:
            download_success = download_all_data(logger)
            state.update("last_download", datetime.now().isoformat())
            if not download_success:
                results["errors"].append("Algunos datos no se descargaron correctamente")
            
            if cleanup:
                logger.info("\n[POST-DOWNLOAD] Limpiando datos antiguos...")
                cleanup_raw_data(logger)
                cleanup_processed_data(logger)
        except Exception as e:
            logger.error(f"Error en descarga: {e}")
            results["errors"].append(str(e))
            results["success"] = False
    
    # Cargar modelo y datos
    if predict or force_finetune:
        try:
            model, model_name = load_model(logger, state)
            df, feature_cols, mean, std = prepare_data(logger)
        except Exception as e:
            logger.error(f"Error cargando modelo/datos: {e}")
            results["errors"].append(str(e))
            results["success"] = False
            return results
    
    # Fine-tuning (si corresponde)
    if force_finetune or (predict and state.should_finetune()):
        try:
            logger.info("\n⚡ Se requiere fine-tuning del modelo")
            model = finetune_model(model, df, feature_cols, logger, state)
            model_name = state.state["current_model"]
            
            df, feature_cols, mean, std = prepare_data(logger)
            
            if cleanup:
                logger.info("\n[POST-FINETUNE] Limpiando modelos antiguos...")
                cleanup_models(logger)
        except Exception as e:
            logger.error(f"Error en fine-tuning: {e}")
            results["errors"].append(str(e))
    
    # Predicción
    if predict:
        try:
            pred_df = predict_future(model, df, feature_cols, mean, std, logger)
            
            csv_path = save_predictions(pred_df, model_name, logger)
            plot_path = plot_predictions(df, pred_df, model_name, logger)
            
            state.update("last_prediction", datetime.now().isoformat())
            state.add_execution("prediction", {
                "model": model_name,
                "csv_path": str(csv_path),
                "plot_path": str(plot_path),
            })
            
            results["predictions"] = pred_df.to_dict("records")
            results["csv_path"] = str(csv_path)
            results["plot_path"] = str(plot_path)
            
            if cleanup:
                logger.info("\n[POST-PREDICTION] Limpiando predicciones antiguas...")
                cleanup_predictions(logger)
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            results["errors"].append(str(e))
            results["success"] = False
    
    # Limpieza final
    if cleanup:
        logger.info("\n[FINAL CLEANUP] Ejecutando limpieza completa...")
        results["cleanup"] = run_full_cleanup(logger)
    
    # Resumen final
    logger.info("\n" + "=" * 70)
    if results["success"] and not results["errors"]:
        logger.info("✓ PIPELINE COMPLETADO EXITOSAMENTE")
    elif results["errors"]:
        logger.warning(f"⚠ PIPELINE COMPLETADO CON ADVERTENCIAS: {len(results['errors'])} errores")
        for err in results["errors"]:
            logger.warning(f"   - {err}")
    else:
        logger.error("✗ PIPELINE FALLIDO")
    logger.info("=" * 70 + "\n")
    
    return results
