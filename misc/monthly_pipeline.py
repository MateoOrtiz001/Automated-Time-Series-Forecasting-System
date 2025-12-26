"""
Pipeline Mensual Automatizado para Predicción de Inflación
============================================================

Este script automatiza:
1. Descarga mensual de datos (BanRep, FAO, Brent)
2. Consolidación de datos procesados
3. Predicción de inflación a 12 meses usando modelo TFT
4. Fine-tuning del modelo cada 3 meses

Estructura de carpetas:
- misc/models/          → Modelos (base y fine-tuned)
- misc/results/         → Predicciones y métricas
- misc/logs/            → Logs de ejecución

Uso:
    python misc/monthly_pipeline.py                    # Ejecución completa
    python misc/monthly_pipeline.py --download-only    # Solo descargar datos
    python misc/monthly_pipeline.py --predict-only     # Solo predicción
    python misc/monthly_pipeline.py --finetune         # Forzar fine-tuning
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para servidores
import matplotlib.pyplot as plt

# Configurar paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

# Imports del proyecto
from src.etl.dataExtractor import (
    extraer_suameca_sin_api,
    extraer_fao_indices,
    extraer_brent_fred,
    consolidar_suameca_json_a_csv_mensual,
)
from src.model.model import TFTModel

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
CONFIG = {
    # Rutas
    "misc_dir": SCRIPT_DIR,
    "models_dir": SCRIPT_DIR / "models",
    "results_dir": SCRIPT_DIR / "results",
    "logs_dir": SCRIPT_DIR / "logs",
    "data_raw_dir": ROOT_DIR / "data" / "raw",
    "data_proc_dir": ROOT_DIR / "data" / "proc",
    
    # Modelo
    "base_model": "tft_base.keras",
    "lookback_steps": 12,
    "forecast_horizon": 1,
    "target_col": "Inflacion_total",
    "future_months": 12,
    
    # Fine-tuning
    "finetune_interval_months": 3,
    "finetune_epochs": 50,
    "finetune_patience": 15,
    "finetune_batch_size": 32,
    "finetune_lr": 5e-4,
    
    # Split ratios
    "val_ratio": 0.1,
    "test_ratio": 0.1,
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
def setup_logging() -> logging.Logger:
    """Configura logging a archivo y consola."""
    log_file = CONFIG["logs_dir"] / f"pipeline_{datetime.now().strftime('%Y%m')}.log"
    
    # Crear directorio si no existe
    CONFIG["logs_dir"].mkdir(parents=True, exist_ok=True)
    
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
        self.state_file = state_file or (CONFIG["misc_dir"] / "pipeline_state.json")
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
# FUNCIONES DE DESCARGA
# ============================================================================
def download_all_data(logger: logging.Logger) -> bool:
    """Descarga todos los datos necesarios."""
    logger.info("="*60)
    logger.info("INICIANDO DESCARGA DE DATOS")
    logger.info("="*60)
    
    success = True
    
    # 1. Datos BanRep (SUAMECA)
    logger.info("\n[1/3] Descargando datos de BanRep (SUAMECA)...")
    try:
        extraer_suameca_sin_api(output_dir=str(CONFIG["data_raw_dir"] / "banrep" / "suameca"))
        logger.info(" BanRep descargado correctamente")
    except Exception as e:
        logger.error(f" Error descargando BanRep: {e}")
        success = False
    
    # 2. Datos FAO
    logger.info("\n[2/3] Descargando índice FAO...")
    try:
        external_dir = CONFIG["data_raw_dir"] / "external"
        external_dir.mkdir(parents=True, exist_ok=True)
        extraer_fao_indices(output_dir=str(external_dir))
        logger.info(" FAO descargado correctamente")
    except Exception as e:
        logger.error(f" Error descargando FAO: {e}")
        success = False
    
    # 3. Datos Brent (FRED)
    logger.info("\n[3/3] Descargando precio Brent (FRED)...")
    try:
        external_dir = CONFIG["data_raw_dir"] / "external"
        extraer_brent_fred(output_dir=str(external_dir))
        logger.info(" Brent descargado correctamente")
    except Exception as e:
        logger.error(f" Error descargando Brent: {e}")
        success = False
    
    # 4. Consolidar datos
    if success:
        logger.info("\n[4/4] Consolidando datos procesados...")
        try:
            consolidar_suameca_json_a_csv_mensual(
                input_dir=str(CONFIG["data_raw_dir"] / "banrep" / "suameca"),
                proc_dir=str(CONFIG["data_proc_dir"]),
                external_dir=str(CONFIG["data_raw_dir"] / "external"),
            )
            logger.info(" Datos consolidados correctamente")
        except Exception as e:
            logger.error(f" Error consolidando datos: {e}")
            success = False
    
    return success


# ============================================================================
# FUNCIONES DE MODELO
# ============================================================================
def load_model(logger: logging.Logger, state: PipelineState) -> Tuple[TFTModel, str]:
    """Carga el modelo más reciente (base o fine-tuned)."""
    models_dir = CONFIG["models_dir"]
    
    # Buscar el modelo más reciente
    current_model = state.state["current_model"]
    model_path = models_dir / current_model
    
    if not model_path.exists():
        # Fallback al modelo base
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


def prepare_data(logger: logging.Logger) -> Tuple[pd.DataFrame, list, np.ndarray, np.ndarray]:
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
    feature_cols: list,
    mean: np.ndarray,
    std: np.ndarray,
    logger: logging.Logger,
    n_months: int = 12,
) -> pd.DataFrame:
    """Realiza predicciones iterativas para los próximos n_months meses."""
    logger.info(f"\nPrediciendo {n_months} meses futuros...")
    
    # Preparar datos
    data_features = df[feature_cols].to_numpy(dtype=np.float32)
    target_idx = feature_cols.index(CONFIG["target_col"])
    
    # Ventana inicial
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
        
        # Actualizar ventana
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
    feature_cols: list,
    logger: logging.Logger,
    state: PipelineState,
) -> TFTModel:
    """Realiza fine-tuning del modelo con los datos más recientes."""
    logger.info("\n" + "="*60)
    logger.info("INICIANDO FINE-TUNING DEL MODELO")
    logger.info("="*60)
    
    # Preparar datos
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
    
    X_train, X_val, X_test, mean, std = TFTModel.standardize_from_train(X_train, X_val, X_test)
    
    logger.info(f"Train: {X_train.shape[0]} samples")
    logger.info(f"Val: {X_val.shape[0] if X_val is not None else 0} samples")
    logger.info(f"Test: {X_test.shape[0] if X_test is not None else 0} samples")
    
    # Nombre del modelo fine-tuned
    timestamp = datetime.now().strftime("%Y%m")
    finetune_count = state.state["finetune_count"] + 1
    new_model_name = f"tft_finetuned_{timestamp}_v{finetune_count}.keras"
    new_model_path = CONFIG["models_dir"] / new_model_name
    
    # Re-compilar con learning rate más bajo
    model.compile(learning_rate=CONFIG["finetune_lr"])
    
    # Fine-tuning
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
    
    # Evaluar
    logger.info("\nEvaluación post fine-tuning:")
    
    for name, X, y_true in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        if X is None or len(X) == 0:
            continue
        preds = model.predict(X)
        y_pred = preds.get("median", preds.get("predictions"))
        y_pred = np.ravel(y_pred)
        if y_pred.size >= 3 * len(y_true):
            # Tiene 3 cuantiles, tomar la mediana
            y_pred = y_pred.reshape(-1, 3)[:, 1]
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        logger.info(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    # Actualizar estado
    state.update("last_finetune", datetime.now().isoformat())
    state.update("finetune_count", finetune_count)
    state.update("current_model", new_model_name)
    
    logger.info(f"\n Modelo guardado: {new_model_name}")
    
    return model


def save_predictions(pred_df: pd.DataFrame, model_name: str, logger: logging.Logger) -> Path:
    """Guarda las predicciones en CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Añadir metadatos
    pred_df = pred_df.copy()
    pred_df["model"] = model_name
    pred_df["generated_at"] = timestamp
    
    # Guardar
    output_path = CONFIG["results_dir"] / f"predictions_{timestamp}.csv"
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    
    logger.info(f" Predicciones guardadas: {output_path.name}")
    
    return output_path


def plot_predictions(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    model_name: str,
    logger: logging.Logger,
) -> Path:
    """Genera gráfico de predicciones."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Datos históricos (últimos 36 meses)
    recent_df = df.tail(36)
    ax.plot(
        recent_df["date"],
        recent_df[CONFIG["target_col"]],
        label="Histórico",
        color="blue",
        linewidth=2,
    )
    
    # Predicciones
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
    
    # Intervalo de confianza
    if pred_df["lower"].notna().any():
        ax.fill_between(
            pred_df["date"],
            pred_df["lower"],
            pred_df["upper"],
            alpha=0.2,
            color="red",
            label="IC 80%",
        )
    
    # Línea vertical de corte
    ax.axvline(x=df["date"].iloc[-1], color="gray", linestyle=":", alpha=0.7)
    
    ax.set_title(f"Predicción de {CONFIG['target_col']} - {datetime.now().strftime('%Y-%m-%d')}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(f"{CONFIG['target_col']} (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = CONFIG["results_dir"] / f"predictions_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f" Gráfico guardado: {plot_path.name}")
    
    return plot_path


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================
def run_pipeline(
    download: bool = True,
    predict: bool = True,
    force_finetune: bool = False,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """Ejecuta el pipeline completo."""
    if logger is None:
        logger = setup_logging()
    
    state = PipelineState()
    results = {"success": True, "errors": []}
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE MENSUAL DE PREDICCIÓN DE INFLACIÓN")
    logger.info("="*70)
    logger.info(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Modelo actual: {state.state['current_model']}")
    logger.info(f"Último fine-tuning: {state.state['last_finetune'] or 'Nunca'}")
    
    # 1. Descarga de datos
    if download:
        try:
            download_success = download_all_data(logger)
            state.update("last_download", datetime.now().isoformat())
            if not download_success:
                results["errors"].append("Algunos datos no se descargaron correctamente")
        except Exception as e:
            logger.error(f"Error en descarga: {e}")
            results["errors"].append(str(e))
            results["success"] = False
    
    # 2. Cargar modelo y datos
    if predict or force_finetune:
        try:
            model, model_name = load_model(logger, state)
            df, feature_cols, mean, std = prepare_data(logger)
        except Exception as e:
            logger.error(f"Error cargando modelo/datos: {e}")
            results["errors"].append(str(e))
            results["success"] = False
            return results
    
    # 3. Fine-tuning (si corresponde)
    if force_finetune or (predict and state.should_finetune()):
        try:
            logger.info("\n Se requiere fine-tuning del modelo")
            model = finetune_model(model, df, feature_cols, logger, state)
            model_name = state.state["current_model"]
            
            # Recargar mean/std después del fine-tuning
            df, feature_cols, mean, std = prepare_data(logger)
        except Exception as e:
            logger.error(f"Error en fine-tuning: {e}")
            results["errors"].append(str(e))
            # Continuar con predicción usando modelo actual
    
    # 4. Predicción
    if predict:
        try:
            pred_df = predict_future(model, df, feature_cols, mean, std, logger)
            
            # Guardar resultados
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
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            results["errors"].append(str(e))
            results["success"] = False
    
    # Resumen final
    logger.info("\n" + "="*70)
    if results["success"] and not results["errors"]:
        logger.info(" PIPELINE COMPLETADO EXITOSAMENTE")
    elif results["errors"]:
        logger.warning(f" PIPELINE COMPLETADO CON ADVERTENCIAS: {len(results['errors'])} errores")
        for err in results["errors"]:
            logger.warning(f"   - {err}")
    else:
        logger.error(" PIPELINE FALLIDO")
    logger.info("="*70 + "\n")
    
    return results


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline mensual para predicción de inflación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python misc/monthly_pipeline.py                    # Ejecución completa
  python misc/monthly_pipeline.py --download-only    # Solo descargar datos
  python misc/monthly_pipeline.py --predict-only     # Solo predicción
  python misc/monthly_pipeline.py --finetune         # Forzar fine-tuning
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
    
    args = parser.parse_args()
    
    # Determinar qué ejecutar
    download = not args.predict_only and not args.no_download
    predict = not args.download_only
    force_finetune = args.finetune
    
    logger = setup_logging()
    
    results = run_pipeline(
        download=download,
        predict=predict,
        force_finetune=force_finetune,
        logger=logger,
    )
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
