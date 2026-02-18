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


# CONFIGURACIÓN GLOBAL
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
    "forecast_horizon": 6,
    "target_col": "Inflacion_total",
    "future_months": 6,
    
    # Fine-tuning
    "finetune_interval_months": 3,
    "finetune_epochs": 50,
    "finetune_patience": 15,
    "finetune_batch_size": 32,
    "finetune_lr": 5e-4,
    "finetune_window_months": 120,  # Ventana móvil: últimos 10 años
    "finetune_val_ratio": 0.1,      # Val del inicio (no del final)
    "finetune_test_ratio": 0.0,     # Sin test en fine-tuning
    
    # Entrenamiento desde cero
    "train_epochs": 100,
    "train_batch_size": 64,
    "train_lr": 1e-3,
    
    # Hiperparámetros del modelo
    "tft_units": 48,
    "tft_heads": 2,
    "tft_lstm_layers": 1,
    "tft_grn_layers": 1,
    "tft_dropout": 0.1,
    
    # Split ratios (solo train/test, sin validación)
    "val_ratio": 0.0,
    "test_ratio": 0.15,
    
    # Sistema de rotación (mantener solo N versiones de cada tipo de archivo)
    "max_data_versions": 2,
    "max_model_versions": 2,
    "max_prediction_versions": 2,
    "max_raw_versions": 2,
    
    # Clasificación de covariables para predicción
    "future_known_cols": ["sin_month", "cos_month"],
    "future_forecast_cols": {
        "IPP": {"method": "holt_damped"},
        "TRM": {"method": "holt_damped"},
        "Brent": {"method": "holt_damped"},
        "FAO": {"method": "holt_damped"},
        "Tasa_interes_colocacion_total": {"method": "ses"},
        "PIB_real_trimestral_2015_AE": {"method": "holt_damped_quarterly"},
    },
    "past_only_cols": [],
    
    # Features futuras para el decoder (orden importa)
    "future_feature_cols": [
        "sin_month", "cos_month", "IPP",
        "TRM", "Brent", "FAO",
        "Tasa_interes_colocacion_total", "PIB_real_trimestral_2015_AE",
    ],
    
    # Pasos de pronóstico para covariables
    "covariate_forecast_steps": 6,
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
    "sin_month": "Componente sinusoidal del mes (estacionalidad)",
    "cos_month": "Componente cosenoidal del mes (estacionalidad)",
}


# FEATURE ENGINEERING
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features de calendario determinísticos (conocidos a futuro).
    
    Añade codificación cíclica sin/cos del mes del año, lo que permite al
    modelo capturar patrones estacionales sin discontinuidades.
    
    Args:
        df: DataFrame con columna 'date' (datetime).
    
    Returns:
        DataFrame con columnas 'sin_month' y 'cos_month' añadidas.
    """
    df = df.copy()
    month = df["date"].dt.month
    df["sin_month"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
    return df


def forecast_arima_ipp(
    series: pd.Series,
    n_steps: int = 6,
    order: tuple = (1, 1, 1),
    logger: logging.Logger = None,
) -> np.ndarray:
    """Genera pronóstico para IPP (covariable futura pronosticada).
    
    Usa Holt exponential smoothing con tendencia amortiguada, adecuado para
    series con tendencia como el IPP (índice de precios que sube gradualmente).
    
    Args:
        series: Serie histórica de IPP.
        n_steps: Número de meses a pronosticar.
        order: Reservado para compatibilidad (no se usa actualmente).
        logger: Logger para registrar operaciones.
    
    Returns:
        Array con n_steps valores pronosticados.
    """
    from statsmodels.tsa.holtwinters import Holt
    
    try:
        model = Holt(series.values.astype(np.float64), damped_trend=True)
        fitted = model.fit()
        forecast = fitted.forecast(n_steps)
        
        if logger:
            logger.info(f"  Holt (damped trend) para IPP: AIC={fitted.aic:.2f}")
            for i, val in enumerate(forecast):
                logger.info(f"    Mes +{i+1}: IPP={val:.2f}")
        
        return np.array(forecast, dtype=np.float32)
    
    except Exception as e:
        if logger:
            logger.warning(f"  Pronóstico IPP falló: {e}. Usando último valor conocido.")
        last_val = float(series.iloc[-1])
        return np.full(n_steps, last_val, dtype=np.float32)


# PRONÓSTICO DE COVARIABLES EXTERNAS
def forecast_covariate(
    series: pd.Series,
    col_name: str,
    n_steps: int = 6,
    logger: logging.Logger = None,
) -> np.ndarray:
    """Genera pronóstico para una covariable usando el método más adecuado.
    
    Métodos por variable:
    - IPP, TRM, Brent, FAO: Holt exponential smoothing con tendencia amortiguada.
    - Tasa_interes_colocacion_total: Simple Exponential Smoothing (tasa sticky).
    - PIB_real_trimestral_2015_AE: Holt damped sobre datos trimestrales únicos,
      expandidos de vuelta a frecuencia mensual.
    
    Args:
        series: Serie histórica de la covariable.
        col_name: Nombre de la columna.
        n_steps: Número de meses a pronosticar.
        logger: Logger para registrar operaciones.
    
    Returns:
        Array con n_steps valores pronosticados.
    """
    cfg = CONFIG.get("future_forecast_cols", {}).get(col_name, {})
    method = cfg.get("method", "holt_damped")
    
    try:
        if method == "holt_damped_quarterly":
            return _forecast_pib_quarterly(series, n_steps, logger)
        elif method == "ses":
            return _forecast_ses(series, n_steps, col_name, logger)
        else:  # holt_damped (default)
            return _forecast_holt_damped(series, n_steps, col_name, logger)
    except Exception as e:
        if logger:
            logger.warning(f"  Pronóstico {col_name} falló: {e}. Usando último valor conocido.")
        return np.full(n_steps, float(series.iloc[-1]), dtype=np.float32)


def _forecast_holt_damped(
    series: pd.Series,
    n_steps: int,
    col_name: str,
    logger: logging.Logger = None,
) -> np.ndarray:
    """Holt exponential smoothing con tendencia amortiguada.
    
    Adecuado para series con tendencia que se espera se amortigüe a largo plazo:
    IPP, TRM, Brent, FAO.
    """
    from statsmodels.tsa.holtwinters import Holt
    
    model = Holt(series.values.astype(np.float64), damped_trend=True)
    fitted = model.fit()
    forecast = fitted.forecast(n_steps)
    
    if logger:
        logger.info(f"  Holt (damped) para {col_name}: AIC={fitted.aic:.2f}")
    
    return np.array(forecast, dtype=np.float32)


def _forecast_ses(
    series: pd.Series,
    n_steps: int,
    col_name: str,
    logger: logging.Logger = None,
) -> np.ndarray:
    """Simple Exponential Smoothing.
    
    Adecuado para tasas de política monetaria (series sticky, sin tendencia marcada).
    La predicción converge al nivel suavizado.
    """
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    
    model = SimpleExpSmoothing(series.values.astype(np.float64))
    fitted = model.fit()
    forecast = fitted.forecast(n_steps)
    
    if logger:
        logger.info(f"  SES para {col_name}: AIC={fitted.aic:.2f}")
    
    return np.array(forecast, dtype=np.float32)


def _forecast_pib_quarterly(
    series: pd.Series,
    n_steps: int,
    logger: logging.Logger = None,
) -> np.ndarray:
    """Pronóstico de PIB (datos trimestrales repetidos mensualmente).
    
    Extrae los valores trimestrales únicos, pronostica con Holt damped,
    y expande de vuelta a frecuencia mensual repitiendo cada valor 3 veces.
    """
    from statsmodels.tsa.holtwinters import Holt
    
    # Extraer valores trimestrales únicos (cambios de valor)
    unique_mask = series.ne(series.shift())
    quarterly = series[unique_mask].reset_index(drop=True)
    
    # Trimestres necesarios para cubrir n_steps meses
    n_quarters = (n_steps + 2) // 3
    
    model = Holt(quarterly.values.astype(np.float64), damped_trend=True)
    fitted = model.fit()
    q_forecast = fitted.forecast(n_quarters)
    
    # Expandir a mensual (repetir cada trimestre 3 veces)
    monthly = np.repeat(q_forecast, 3)[:n_steps]
    
    if logger:
        logger.info(f"  Holt (damped, quarterly→monthly) para PIB: AIC={fitted.aic:.2f}")
    
    return monthly.astype(np.float32)


def forecast_all_covariates(
    df: pd.DataFrame,
    n_steps: int = None,
    logger: logging.Logger = None,
) -> Dict[str, np.ndarray]:
    """Pronostica todas las covariables futuras configuradas.
    
    Itera sobre CONFIG['future_forecast_cols'] y genera un pronóstico
    para cada covariable usando el método configurado.
    
    Args:
        df: DataFrame con datos históricos.
        n_steps: Meses a pronosticar. Por defecto usa CONFIG['covariate_forecast_steps'].
        logger: Logger.
    
    Returns:
        Dict {nombre_covariable: array de pronósticos}.
    """
    n_steps = n_steps or CONFIG["covariate_forecast_steps"]
    forecasts = {}
    
    forecast_cols = CONFIG.get("future_forecast_cols", {})
    
    if logger:
        logger.info(
            f"\n[COVARIATE FORECASTS] Pronosticando {len(forecast_cols)} "
            f"covariables para {n_steps} meses..."
        )
    
    for col_name in forecast_cols:
        if col_name not in df.columns:
            if logger:
                logger.warning(f"  ⚠ Columna '{col_name}' no encontrada en datos. Saltando.")
            continue
        
        forecast = forecast_covariate(df[col_name], col_name, n_steps, logger)
        forecasts[col_name] = forecast
        
        if logger:
            for i, val in enumerate(forecast):
                logger.info(f"    Mes +{i+1}: {col_name}={val:.2f}")
    
    return forecasts


# PERSISTENCIA DE ESTADÍSTICAS DE ESTANDARIZACIÓN
SCALER_STATS_FILENAME = "scaler_stats.npz"


def save_scaler_stats(
    mean: np.ndarray,
    std: np.ndarray,
    mean_f: np.ndarray = None,
    std_f: np.ndarray = None,
    logger: logging.Logger = None,
) -> None:
    """Guarda las estadísticas de estandarización (mean/std) en disco.
    
    Se guardan junto al modelo para garantizar que la misma normalización
    usada en entrenamiento se aplique durante la predicción.
    
    Se guarda en dos ubicaciones:
    - CONFIG['models_dir'] (models/) para el modelo base.
    - CONFIG['pipeline_models_dir'] (misc/models/) para fine-tuning y deploy.
    """
    data = {"mean": mean, "std": std}
    if mean_f is not None:
        data["mean_f"] = mean_f
    if std_f is not None:
        data["std_f"] = std_f
    
    for save_dir in [CONFIG["models_dir"], CONFIG["pipeline_models_dir"]]:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / SCALER_STATS_FILENAME
        np.savez(path, **data)
    
    if logger:
        logger.info(f"  Scaler stats guardadas en: models/ y misc/models/")


def load_scaler_stats(
    logger: logging.Logger = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Carga las estadísticas de estandarización guardadas.
    
    Busca en: misc/models/ → models/ (en ese orden de prioridad).
    
    Returns:
        (mean, std, mean_f, std_f). mean_f y std_f pueden ser None.
    
    Raises:
        FileNotFoundError: Si no se encuentra el archivo en ninguna ubicación.
    """
    search_dirs = [
        CONFIG["pipeline_models_dir"],
        CONFIG["models_dir"],
    ]
    
    for d in search_dirs:
        path = d / SCALER_STATS_FILENAME
        if path.exists():
            data = np.load(path)
            mean = data["mean"]
            std = data["std"]
            mean_f = data["mean_f"] if "mean_f" in data else None
            std_f = data["std_f"] if "std_f" in data else None
            if logger:
                logger.info(f"  Scaler stats cargadas desde: {path}")
            return mean, std, mean_f, std_f
    
    raise FileNotFoundError(
        f"No se encontró {SCALER_STATS_FILENAME} en: "
        + ", ".join(str(d) for d in search_dirs)
    )


def get_scaler_stats(
    df: pd.DataFrame,
    feature_cols: List[str],
    logger: logging.Logger = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Obtiene las estadísticas de estandarización.
    
    Intenta cargar stats guardadas primero (generadas durante entrenamiento).
    Si no existen, las calcula del split actual y las guarda.
    
    Args:
        df: DataFrame con datos históricos (con calendar features).
        feature_cols: Lista de columnas de features.
        logger: Logger (puede ser None).
    
    Returns:
        (mean, std, mean_f, std_f).
    """
    try:
        return load_scaler_stats(logger)
    except FileNotFoundError:
        if logger:
            logger.warning("  Stats no encontradas en disco. Calculando desde datos actuales...")
        
        future_feature_cols = CONFIG.get("future_feature_cols", [])
        X, _, X_future = TFTModel.make_supervised_windows(
            df=df,
            feature_cols=feature_cols,
            target_col=CONFIG["target_col"],
            lookback_steps=CONFIG["lookback_steps"],
            forecast_horizon=CONFIG["forecast_horizon"],
            future_feature_cols=future_feature_cols if future_feature_cols else None,
        )
        
        n = len(X)
        n_test = int(n * CONFIG["test_ratio"])
        n_train = n - n_test
        
        _, _, _, mean, std = TFTModel.standardize_from_train(X[:n_train])
        
        mean_f, std_f = None, None
        if X_future is not None:
            _, _, _, mean_f, std_f = TFTModel.standardize_from_train(X_future[:n_train])
        
        save_scaler_stats(mean, std, mean_f, std_f, logger=logger)
        
        return mean, std, mean_f, std_f


# LOGGING
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


# ESTADO DEL PIPELINE
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


# LIMPIEZA DE ARCHIVOS
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
    """Limpia archivos de predicciones antiguos, dejando solo *_latest.*"""
    deleted = []
    
    results_dir = CONFIG["pipeline_results_dir"]
    if not results_dir.exists():
        return deleted
    
    csv_files = [f for f in results_dir.glob("predictions_*.csv") 
                 if f.name != "predictions_latest.csv"]
    
    for f in csv_files:
        try:
            f.unlink()
            deleted.append(f)
            logger.info(f"   [CLEANUP] Eliminado: {f.name}")
        except Exception as e:
            logger.warning(f"   [CLEANUP] No se pudo eliminar {f.name}: {e}")
     
    png_files = [f for f in results_dir.glob("predictions_plot_*.png")
                 if f.name != "predictions_plot_latest.png"]
    
    for f in png_files:
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


# DESCARGA DE DATOS
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


# FUNCIONES DE MODELO
def load_model(logger: logging.Logger, state: PipelineState) -> Tuple[TFTModel, str]:
    """Carga el modelo más reciente (base o fine-tuned)."""
    models_dir = CONFIG["pipeline_models_dir"]
    root_models_dir = CONFIG["models_dir"]

    current_model = state.state["current_model"]
    
    # Cargar datos para obtener n_features (incluye calendar features)
    df = TFTModel.load_latest_proc_csv(CONFIG["data_proc_dir"])
    df = df.dropna().reset_index(drop=True)
    df = add_calendar_features(df)
    feature_cols = [c for c in df.columns if c != "date"]
    n_features = len(feature_cols)
    future_feature_cols = CONFIG.get("future_feature_cols", [])
    
    # Crear instancia del modelo
    tft = TFTModel(
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
        n_features=n_features,
        n_future_features=len(future_feature_cols),
        units=CONFIG["tft_units"],
        num_heads=CONFIG["tft_heads"],
        num_lstm_layers=CONFIG["tft_lstm_layers"],
        num_grn_layers=CONFIG["tft_grn_layers"],
        dropout_rate=CONFIG["tft_dropout"],
        num_quantiles=3,
    )
    
    # Probar candidatos en orden de preferencia con fallback robusto
    candidate_paths: List[Path] = []
    for candidate in [
        models_dir / current_model,
        root_models_dir / current_model,
        models_dir / CONFIG["base_model"],
        root_models_dir / "tft_best.keras",
    ]:
        if candidate.exists() and candidate not in candidate_paths:
            candidate_paths.append(candidate)

    if not candidate_paths:
        raise FileNotFoundError(
            "No se encontró ningún modelo utilizable en "
            f"{models_dir} ni {root_models_dir}"
        )

    load_errors = []
    for model_path in candidate_paths:
        try:
            logger.info(f"Cargando modelo: {model_path.name}")
            tft.build_model()
            tft.model.load_weights(str(model_path))

            if state.state.get("current_model") != model_path.name:
                state.update("current_model", model_path.name)
                logger.info(
                    f"Modelo actual ajustado automáticamente a: {model_path.name}"
                )

            return tft, model_path.name
        except Exception as exc:
            load_errors.append(f"{model_path.name}: {exc}")
            logger.warning(
                f"No se pudo cargar {model_path.name}; probando fallback."
            )

    raise RuntimeError(
        "No fue posible cargar ningún modelo compatible. "
        f"Errores: {' | '.join(load_errors)}"
    )


def prepare_data(logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """Prepara los datos para predicción/entrenamiento.
    
    Carga el CSV procesado más reciente, elimina NaN, y agrega features
    de calendario (sin_month, cos_month).
    
    Las estadísticas de estandarización se obtienen por separado
    mediante get_scaler_stats() o se computan durante el entrenamiento.
    
    Returns:
        Tupla (df, feature_cols).
    """
    df = TFTModel.load_latest_proc_csv(CONFIG["data_proc_dir"])
    df = df.dropna().reset_index(drop=True)
    
    # Agregar features de calendario (conocidos a futuro)
    df = add_calendar_features(df)
    
    feature_cols = [c for c in df.columns if c != "date"]
    future_feature_cols = CONFIG.get("future_feature_cols", [])
    
    logger.info(f"Dataset cargado: {len(df)} filas")
    logger.info(f"Rango: {df['date'].min()} → {df['date'].max()}")
    logger.info(f"Features pasadas ({len(feature_cols)}):")
    for col in feature_cols:
        desc = VARIABLE_DESCRIPTIONS.get(col, col)
        logger.info(f"   • {col}: {desc}")
    if future_feature_cols:
        logger.info(f"Features futuras (decoder): {future_feature_cols}")
    
    return df, feature_cols


def predict_future(
    model: TFTModel,
    df: pd.DataFrame,
    feature_cols: List,
    mean: np.ndarray,
    std: np.ndarray,
    logger: logging.Logger,
    n_months: int = None,
    covariate_forecasts: Dict[str, np.ndarray] = None,
    mean_f: np.ndarray = None,
    std_f: np.ndarray = None,
) -> pd.DataFrame:
    """Realiza predicción directa de N meses con decoder de covariables futuras.
    
    El modelo recibe dos entradas:
    - past_inputs: ventana de lookback con todas las features estandarizadas.
    - future_inputs: covariables futuras conocidas/pronosticadas estandarizadas
      con estadísticas guardadas del entrenamiento.
    
    Args:
        model: Modelo TFT cargado (con decoder).
        df: DataFrame con datos históricos (ya con calendar features).
        feature_cols: Lista de columnas de features (en orden del modelo).
        mean: Media para estandarización del lookback (guardada del train).
        std: Desviación estándar del lookback (guardada del train).
        logger: Logger.
        n_months: Meses a predecir. Por defecto usa CONFIG['future_months'].
        covariate_forecasts: Dict con pronósticos de covariables {nombre: array}.
        mean_f: Media para estandarización de features futuras (guardada del train).
        std_f: Desviación estándar de features futuras (guardada del train).
    """
    n_months = n_months or CONFIG["future_months"]
    future_feature_cols = CONFIG.get("future_feature_cols", [])
    covariate_forecasts = covariate_forecasts or {}
    
    logger.info(f"\nPrediciendo {n_months} meses futuros (ventana fija)...")
    logger.info(f"  Features futuras (decoder): {future_feature_cols}")
    
    # Preparar entrada del encoder (lookback)
    data_features = df[feature_cols].to_numpy(dtype=np.float32)
    current_window = data_features[-CONFIG["lookback_steps"]:].copy()
    current_window_std = ((current_window - mean) / std).astype(np.float32)
    X_past = current_window_std[np.newaxis, :, :]  # (1, lookback, n_features)
    
    # Preparar entrada del decoder (features futuras)
    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=n_months,
        freq="MS",
    )
    
    # Construir features futuras
    X_future = np.zeros((1, n_months, len(future_feature_cols)), dtype=np.float32)
    
    for i, date in enumerate(future_dates):
        month = date.month
        for j, col in enumerate(future_feature_cols):
            if col == "sin_month":
                X_future[0, i, j] = np.sin(2 * np.pi * month / 12)
            elif col == "cos_month":
                X_future[0, i, j] = np.cos(2 * np.pi * month / 12)
            elif col in covariate_forecasts and i < len(covariate_forecasts[col]):
                X_future[0, i, j] = covariate_forecasts[col][i]
            else:
                # Fallback: último valor conocido
                X_future[0, i, j] = float(df[col].iloc[-1])
    
    # Estandarizar features futuras
    if mean_f is not None and std_f is not None:
        X_future = ((X_future - mean_f) / std_f).astype(np.float32)
    
    # Predicción directa (una sola pasada por el modelo)
    pred_result = model.predict([X_past, X_future])
    
    # Extraer predicciones: (1, horizon, quantiles) → (horizon,)
    median = np.ravel(pred_result.get("median", pred_result["predictions"][..., 1]))
    lower = np.ravel(pred_result["lower"]) if "lower" in pred_result else [None] * n_months
    upper = np.ravel(pred_result["upper"]) if "upper" in pred_result else [None] * n_months
    
    predictions = []
    for i in range(n_months):
        predictions.append({
            "date": future_dates[i],
            "prediction": float(median[i]),
            "lower": float(lower[i]) if lower[i] is not None else None,
            "upper": float(upper[i]) if upper[i] is not None else None,
        })
    
    pred_df = pd.DataFrame(predictions)
    
    logger.info(f"\nPredicciones ({CONFIG['target_col']}) — ventana fija de {n_months} meses:")
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
    - Sin early stopping: entrena por épocas fijas.
    - Todos los datos de la ventana se usan para entrenamiento.
    """
    logger.info("\n" + "=" * 60)
    logger.info("INICIANDO FINE-TUNING DEL MODELO")
    logger.info("=" * 60)
    
    # --- Ventana móvil: recortar a los últimos N meses ---
    window_months = CONFIG.get("finetune_window_months", 120)
    original_len = len(df)
    
    if len(df) > window_months:
        df = df.iloc[-window_months:].reset_index(drop=True)
        logger.info(f"Ventana móvil: usando últimos {window_months} meses ({len(df)} filas de {original_len} originales)")
    else:
        logger.info(f"Usando todos los datos disponibles ({len(df)} meses, menor que ventana de {window_months})")
    
    X, y, X_future = TFTModel.make_supervised_windows(
        df=df,
        feature_cols=feature_cols,
        target_col=CONFIG["target_col"],
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
        future_feature_cols=CONFIG.get("future_feature_cols"),
    )
    
    # Sin split: todos los datos para entrenamiento (sin early stopping)
    X_train, y_train = X, y
    
    # Estandarizar features pasadas
    X_train_s, _, _, mean, std = TFTModel.standardize_from_train(X_train, None, None)
    
    # Estandarizar features futuras
    train_input = X_train_s
    mean_f, std_f = None, None
    if X_future is not None:
        Xf_train_s, _, _, mean_f, std_f = TFTModel.standardize_from_train(
            X_future, None, None
        )
        train_input = [X_train_s, Xf_train_s]
    
    # Guardar estadísticas de estandarización
    save_scaler_stats(mean, std, mean_f, std_f, logger=logger)
    
    logger.info(f"Train: {len(X_train)} samples (todos los datos de la ventana)")
    logger.info(f"Early stopping: deshabilitado")
    
    timestamp = datetime.now().strftime("%Y%m")
    finetune_count = state.state["finetune_count"] + 1
    new_model_name = f"tft_finetuned_{timestamp}_v{finetune_count}.keras"
    new_model_path = CONFIG["pipeline_models_dir"] / new_model_name
    
    CONFIG["pipeline_models_dir"].mkdir(parents=True, exist_ok=True)
    
    model.compile(learning_rate=CONFIG["finetune_lr"])
    
    logger.info(f"\nEntrenando {CONFIG['finetune_epochs']} épocas con lr={CONFIG['finetune_lr']}...")
    
    history = model.fit(
        X_train=train_input,
        y_train=y_train,
        X_val=None,
        y_val=None,
        epochs=CONFIG["finetune_epochs"],
        batch_size=CONFIG["finetune_batch_size"],
        patience=CONFIG["finetune_patience"],
        save_best=True,
        model_path=str(new_model_path),
    )
    
    # Evaluación in-sample
    logger.info("\nEvaluación post fine-tuning (in-sample):")
    preds = model.predict(train_input)
    y_pred = preds.get("median", preds.get("predictions"))
    y_pred = np.ravel(y_pred)
    y_true_flat = np.ravel(y_train)
    if y_pred.size > y_true_flat.size:
        y_pred = y_pred[:y_true_flat.size]
    
    mae = np.mean(np.abs(y_true_flat - y_pred))
    rmse = np.sqrt(np.mean((y_true_flat - y_pred) ** 2))
    logger.info(f"  Train: MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    state.update("last_finetune", datetime.now().isoformat())
    state.update("finetune_count", finetune_count)
    state.update("current_model", new_model_name)
    
    logger.info(f"\n✓ Modelo guardado: {new_model_name}")
    
    return model


def train_model_from_scratch(
    logger: logging.Logger,
    epochs: int = None,
) -> Tuple[TFTModel, str]:
    """Entrena un modelo TFT desde cero con decoder para covariables futuras.
    
    Configuración:
    - Split 85/15 train/test (sin validación).
    - Sin early stopping.
    - Todas las covariables futuras alimentan el decoder.
    - 100 épocas, units=48, heads=2, 1 capa GRN.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ENTRENAMIENTO DE MODELO DESDE CERO")
    logger.info("=" * 60)
    
    epochs = epochs or CONFIG["train_epochs"]
    models_dir = CONFIG["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar y preparar datos
    df, feature_cols = prepare_data(logger)
    future_feature_cols = CONFIG.get("future_feature_cols", [])
    
    X, y, X_future = TFTModel.make_supervised_windows(
        df=df,
        feature_cols=feature_cols,
        target_col=CONFIG["target_col"],
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
        future_feature_cols=future_feature_cols if future_feature_cols else None,
    )
    
    # Split 85/15 (sin validación)
    n = len(X)
    n_test = int(n * CONFIG["test_ratio"])
    n_train = n - n_test
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    
    # Estandarizar features pasadas
    X_train_s, _, X_test_s, mean, std = TFTModel.standardize_from_train(X_train, None, X_test)
    
    # Preparar inputs (con features futuras si hay decoder)
    train_input = X_train_s
    test_input = X_test_s
    
    mean_f, std_f = None, None
    if X_future is not None:
        Xf_train, Xf_test = X_future[:n_train], X_future[n_train:]
        Xf_train_s, _, Xf_test_s, mean_f, std_f = TFTModel.standardize_from_train(Xf_train, None, Xf_test)
        train_input = [X_train_s, Xf_train_s]
        test_input = [X_test_s, Xf_test_s]
    
    # Guardar estadísticas de estandarización
    save_scaler_stats(mean, std, mean_f, std_f, logger=logger)
    
    logger.info(f"Train: {n_train}, Test: {n_test} (split {100-int(CONFIG['test_ratio']*100)}/{int(CONFIG['test_ratio']*100)})")
    logger.info(f"Scaler stats guardadas para uso en predicción.")
    logger.info(f"Validación: deshabilitada")
    logger.info(f"Early stopping: deshabilitado")
    
    # Crear y entrenar modelo
    tft = TFTModel(
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
        n_features=len(feature_cols),
        n_future_features=len(future_feature_cols) if future_feature_cols else 0,
        units=CONFIG["tft_units"],
        num_heads=CONFIG["tft_heads"],
        num_lstm_layers=CONFIG["tft_lstm_layers"],
        num_grn_layers=CONFIG["tft_grn_layers"],
        dropout_rate=CONFIG["tft_dropout"],
        num_quantiles=3,
    )
    
    tft.compile(learning_rate=CONFIG["train_lr"])
    
    model_path = models_dir / "tft_best.keras"
    tft.fit(
        X_train=train_input,
        y_train=y_train,
        X_val=None,
        y_val=None,
        epochs=epochs,
        batch_size=CONFIG["train_batch_size"],
        save_best=True,
        model_path=str(model_path),
    )
    
    # Evaluación en test
    logger.info("\nEvaluación en test set:")
    preds = tft.predict(test_input)
    y_pred = preds.get("median", preds.get("predictions"))
    y_pred = np.ravel(y_pred)
    y_test_flat = np.ravel(y_test)
    if y_pred.size > y_test_flat.size:
        y_pred = y_pred[:y_test_flat.size]
    
    mae = np.mean(np.abs(y_test_flat - y_pred))
    rmse = np.sqrt(np.mean((y_test_flat - y_pred) ** 2))
    logger.info(f"  Test MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    logger.info(f"✓ Modelo guardado: {model_path}")
    
    return tft, str(model_path)


# GUARDADO DE RESULTADOS
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


# PIPELINE PRINCIPAL
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
            df, feature_cols = prepare_data(logger)
            mean, std, mean_f, std_f = get_scaler_stats(df, feature_cols, logger)
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
            
            df, feature_cols = prepare_data(logger)
            mean, std, mean_f, std_f = get_scaler_stats(df, feature_cols, logger)
            
            if cleanup:
                logger.info("\n[POST-FINETUNE] Limpiando modelos antiguos...")
                cleanup_models(logger)
        except Exception as e:
            logger.error(f"Error en fine-tuning: {e}")
            results["errors"].append(str(e))
    
    # Predicción
    if predict:
        try:
            # Pronosticar todas las covariables futuras
            covariate_forecasts = forecast_all_covariates(df, logger=logger)
            
            pred_df = predict_future(
                model, df, feature_cols, mean, std, logger,
                covariate_forecasts=covariate_forecasts,
                mean_f=mean_f,
                std_f=std_f,
            )
            
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
