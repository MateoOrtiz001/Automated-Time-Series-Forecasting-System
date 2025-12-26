"""
Script para entrenar modelos TFT y LSTM, evaluar sobreajuste y realizar predicciones a 12 meses.

Variables soportadas:
- Banrep (Colombia): Inflación, PIB, TRM, Tasa interés, IPP
- Externas: FAO Food Price Index, Brent Oil
"""
import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path (test.py está en la raíz del proyecto)
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.model.model import TFTModel, LSTMModel


# ============================================================================
# CONFIGURACIÓN
# ============================================================================
CONFIG = {
    "lookback_steps": 12,           # 12 meses de historia
    "forecast_horizon": 1,          # Predicción paso a paso
    "target_col": "Inflacion_total",
    "epochs": 200,
    "batch_size": 64,
    "patience": 40,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "learning_rate": 1e-3,
    "future_months": 12,            # Predicción a 12 meses futuros
}

# Directorios
PROC_DIR = ROOT_DIR / "data" / "proc"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Crear directorios si no existen
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Descripciones de variables para interpretación
VARIABLE_DESCRIPTIONS = {
    "Inflacion_total": "Inflación total (variación anual %)",
    "IPP": "Índice de Precios del Productor",
    "PIB_real_trimestral_2015_AE": "PIB real trimestral",
    "Tasa_interes_colocacion_total": "Tasa interés colocación",
    "TRM": "Tasa cambio COP/USD",
    "Brent": "Precio petróleo Brent (USD/barril)",
    "FAO": "Índice precios alimentos FAO",
}


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
def load_and_prepare_data():
    """Carga y prepara los datos del CSV más reciente."""
    df = TFTModel.load_latest_proc_csv(PROC_DIR)
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = [c for c in df.columns if c != "date"]
    
    print(f"Dataset cargado: {len(df)} filas")
    print(f"Rango de fechas: {df['date'].min()} a {df['date'].max()}")
    print(f"Target: {CONFIG['target_col']}")
    print(f"\nFeatures ({len(feature_cols)}):")
    for col in feature_cols:
        desc = VARIABLE_DESCRIPTIONS.get(col, col)
        print(f"   • {col}: {desc}")
    
    return df, feature_cols


def prepare_datasets(df, feature_cols):
    """Prepara los datasets de train/val/test."""
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
        test_ratio=CONFIG["test_ratio"]
    )
    
    # Estandarizar (manejar casos donde val o test son None/vacíos)
    X_val_input = X_val if X_val is not None and len(X_val) > 0 else None
    X_test_input = X_test if X_test is not None and len(X_test) > 0 else None
    
    X_train, X_val_out, X_test_out, mean, std = TFTModel.standardize_from_train(
        X_train, X_val_input, X_test_input
    )
    
    # Actualizar arrays si fueron estandarizados
    if X_val_out is not None:
        X_val = X_val_out
    if X_test_out is not None:
        X_test = X_test_out
    
    print(f"\nShapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    if X_val is not None and len(X_val) > 0:
        print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    else:
        print(f"  X_val: None (sin datos de validación)")
    if X_test is not None and len(X_test) > 0:
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    else:
        print(f"  X_test: None (sin datos de test)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, mean, std


def prepare_full_dataset_for_overfit_check(df, feature_cols, mean, std):
    """Prepara todo el dataset para verificar sobreajuste."""
    X_full, y_full = TFTModel.make_supervised_windows(
        df=df,
        feature_cols=feature_cols,
        target_col=CONFIG["target_col"],
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
    )
    
    # Estandarizar con los mismos parámetros del train
    X_full_std = ((X_full - mean) / std).astype(np.float32)
    
    return X_full_std, y_full


def get_target_dates(df: pd.DataFrame, lookback_steps: int, forecast_horizon: int = 1) -> pd.Series:
    """Fechas alineadas con y al generar ventanas supervisadas.

    Para forecast_horizon=1, y[t] corresponde a df['date'][lookback_steps + t].
    """
    start = lookback_steps + forecast_horizon - 1
    return df["date"].iloc[start:].reset_index(drop=True)


def calculate_metrics(y_true, y_pred):
    """Calcula métricas de evaluación."""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


def train_tft_model(X_train, y_train, X_val, y_val, n_features):
    """Entrena el modelo TFT."""
    print("\n" + "="*60)
    print("ENTRENANDO MODELO TFT (Temporal Fusion Transformer)")
    print("="*60)
    
    # Verificar si hay datos de validación
    has_val = X_val is not None and len(X_val) > 0
    
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
    
    tft.compile(learning_rate=CONFIG["learning_rate"])
    tft.summary()
    
    history = tft.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val if has_val else None,
        y_val=y_val if has_val else None,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        patience=CONFIG["patience"],
        save_best=True,
        model_path=str(MODELS_DIR / "tft_best.keras"),
    )
    
    return tft, history


def train_lstm_model(X_train, y_train, X_val, y_val, n_features):
    """Entrena el modelo LSTM."""
    print("\n" + "="*60)
    print("ENTRENANDO MODELO LSTM")
    print("="*60)
    
    # Verificar si hay datos de validación
    has_val = X_val is not None and len(X_val) > 0
    
    lstm = LSTMModel(
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
        n_features=n_features,
        units=64,
        num_lstm_layers=2,
        dropout_rate=0.1,
    )
    
    lstm.compile(learning_rate=CONFIG["learning_rate"])
    lstm.summary()
    
    history = lstm.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val if has_val else None,
        y_val=y_val if has_val else None,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        patience=CONFIG["patience"],
        save_best=True,
        model_path=str(MODELS_DIR / "lstm_best.keras"),
    )
    
    return lstm, history


def evaluate_overfitting(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evalúa el sobreajuste del modelo."""
    print(f"\n{'='*60}")
    print(f"EVALUACIÓN DE SOBREAJUSTE - {model_name}")
    print("="*60)
    
    results = {}
    
    # Predicciones en cada conjunto (solo si tienen datos)
    datasets = [("Train", X_train, y_train)]
    if X_val is not None and len(X_val) > 0:
        datasets.append(("Validation", X_val, y_val))
    if X_test is not None and len(X_test) > 0:
        datasets.append(("Test", X_test, y_test))
    
    for name, X, y in datasets:
        if model_name == "TFT":
            preds = model.predict(X)
            y_pred = preds.get("median", preds.get("predictions", preds.get("point")))
            if len(y_pred.shape) > 1 and y_pred.shape[-1] == 3:
                y_pred = y_pred[:, 1]  # mediana
        else:
            y_pred = model.predict(X).flatten()
        
        metrics = calculate_metrics(y, y_pred)
        results[name] = metrics
        
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Indicador de sobreajuste (usar val si existe, sino test)
    train_mae = results["Train"]["MAE"]
    
    if "Validation" in results:
        compare_mae = results["Validation"]["MAE"]
        compare_name = "Validation"
    elif "Test" in results:
        compare_mae = results["Test"]["MAE"]
        compare_name = "Test"
    else:
        print("\nNo hay datos de Validation/Test para comparar sobreajuste.")
        return results
    
    overfit_ratio = compare_mae / (train_mae + 1e-8)
    print(f"\nRatio {compare_name}/Train MAE: {overfit_ratio:.2f}")
    if overfit_ratio > 1.5:
        print("⚠️  ALERTA: Posible sobreajuste detectado")
    else:
        print("✓ El modelo generaliza razonablemente bien")
    
    return results


def predict_future(model, model_name, df, feature_cols, mean, std, n_months=12):
    """
    Realiza predicciones iterativas para los próximos n_months meses.
    """
    print(f"\n{'='*60}")
    print(f"PREDICCIÓN A {n_months} MESES - {model_name}")
    print("="*60)
    
    # Preparar los últimos lookback_steps para empezar
    data_features = df[feature_cols].to_numpy(dtype=np.float32)
    
    # Índice del target en las features
    target_idx = feature_cols.index(CONFIG["target_col"])
    
    # Ventana inicial (los últimos lookback_steps)
    current_window = data_features[-CONFIG["lookback_steps"]:].copy()
    
    # Estandarizar
    current_window_std = ((current_window - mean) / std).astype(np.float32)
    
    predictions = []
    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=n_months,
        freq="MS"
    )
    
    for i in range(n_months):
        # Preparar input
        X_input = current_window_std[np.newaxis, :, :]  # (1, lookback, features)
        
        # Predecir
        if model_name == "TFT":
            pred_result = model.predict(X_input)
            pred_value = pred_result.get("median", pred_result.get("predictions"))
            pred_arr = np.ravel(pred_value)
            # Si vienen cuantiles [q10, q50, q90], usar q50
            pred_value = float(pred_arr[1] if pred_arr.size >= 3 else pred_arr[0])
            
            # Guardar también intervalos si están disponibles
            lower = float(np.ravel(pred_result["lower"])[0]) if "lower" in pred_result else None
            upper = float(np.ravel(pred_result["upper"])[0]) if "upper" in pred_result else None
        else:
            pred_value = float(model.predict(X_input).flatten()[0])
            lower, upper = None, None
        
        predictions.append({
            "date": future_dates[i],
            "prediction": pred_value,
            "lower": lower,
            "upper": upper,
        })
        
        # Actualizar ventana para siguiente predicción
        # Shift y añadir nueva predicción (usamos el último valor conocido para otras features)
        new_row = current_window[-1].copy()
        new_row[target_idx] = pred_value  # Actualizar target con predicción
        
        current_window = np.vstack([current_window[1:], new_row])
        current_window_std = ((current_window - mean) / std).astype(np.float32)
    
    # Crear DataFrame con predicciones
    pred_df = pd.DataFrame(predictions)
    
    print(f"\nPredicciones futuras ({CONFIG['target_col']}):")
    print(pred_df.to_string(index=False))
    
    return pred_df


def plot_results(test_dates, y_test, tft_model, lstm_model, X_test, tft_future_preds, lstm_future_preds):
    """Genera gráficos de resultados (solo Test + 12 meses posteriores)."""
    if X_test is None or len(X_test) == 0:
        print("No hay conjunto de prueba para graficar.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

    # --- Plot 1: TFT (Test + Futuro) ---
    ax1 = axes[0]
    tft_preds = tft_model.predict(X_test)
    tft_y_pred = tft_preds.get("median", tft_preds.get("point", tft_preds.get("predictions")))
    tft_y_pred = np.ravel(tft_y_pred)

    ax1.plot(test_dates, y_test, label="Test Real", color="blue", alpha=0.7)
    ax1.plot(test_dates, tft_y_pred, label="TFT Test Pred", color="red", alpha=0.8)
    ax1.plot(tft_future_preds["date"], tft_future_preds["prediction"], label="TFT +12m", color="red", linestyle="--")
    if "lower" in tft_preds and "upper" in tft_preds:
        lower_test = np.ravel(tft_preds["lower"])
        upper_test = np.ravel(tft_preds["upper"])
        if len(lower_test) == len(test_dates) and len(upper_test) == len(test_dates):
            ax1.fill_between(test_dates, lower_test, upper_test, alpha=0.15, color="red")
    if tft_future_preds["lower"].notna().any():
        ax1.fill_between(tft_future_preds["date"], tft_future_preds["lower"], tft_future_preds["upper"], alpha=0.2, color="red")
    ax1.axvline(x=test_dates.iloc[-1], color="gray", linestyle=":", label="Fin Test / Inicio Forecast")
    ax1.set_title(f"TFT - Test + {CONFIG['future_months']} Meses ({CONFIG['target_col']})")
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel(CONFIG["target_col"])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: LSTM (Test + Futuro) ---
    ax2 = axes[1]
    lstm_preds = lstm_model.predict(X_test).flatten()
    ax2.plot(test_dates, y_test, label="Test Real", color="blue", alpha=0.7)
    ax2.plot(test_dates, lstm_preds, label="LSTM Test Pred", color="green", alpha=0.8)
    ax2.plot(lstm_future_preds["date"], lstm_future_preds["prediction"], label="LSTM +12m", color="green", linestyle="--")
    ax2.axvline(x=test_dates.iloc[-1], color="gray", linestyle=":", label="Fin Test / Inicio Forecast")
    ax2.set_title(f"LSTM - Test + {CONFIG['future_months']} Meses ({CONFIG['target_col']})")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel(CONFIG["target_col"])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    plot_path = RESULTS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nGráfico guardado en: {plot_path}")
    
    plt.show()


def plot_training_history(tft_history, lstm_history):
    """Grafica el historial de entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # TFT
    ax1 = axes[0]
    ax1.plot(tft_history.history["loss"], label="Train Loss")
    if "val_loss" in tft_history.history:
        ax1.plot(tft_history.history["val_loss"], label="Val Loss")
    ax1.set_title("TFT - Historial de Entrenamiento")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # LSTM
    ax2 = axes[1]
    ax2.plot(lstm_history.history["loss"], label="Train Loss")
    if "val_loss" in lstm_history.history:
        ax2.plot(lstm_history.history["val_loss"], label="Val Loss")
    ax2.set_title("LSTM - Historial de Entrenamiento")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Historial de entrenamiento guardado en: {plot_path}")
    
    plt.show()


def save_results(tft_results, lstm_results, tft_future, lstm_future):
    """Guarda los resultados en CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Métricas de sobreajuste
    metrics_data = []
    for model_name, results in [("TFT", tft_results), ("LSTM", lstm_results)]:
        for dataset, metrics in results.items():
            for metric, value in metrics.items():
                metrics_data.append({
                    "Model": model_name,
                    "Dataset": dataset,
                    "Metric": metric,
                    "Value": value,
                })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = RESULTS_DIR / f"metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Métricas guardadas en: {metrics_path}")
    
    # Predicciones futuras
    tft_future["model"] = "TFT"
    lstm_future["model"] = "LSTM"
    future_df = pd.concat([tft_future, lstm_future], ignore_index=True)
    future_path = RESULTS_DIR / f"future_predictions_{timestamp}.csv"
    future_df.to_csv(future_path, index=False)
    print(f"Predicciones futuras guardadas en: {future_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*70)
    print("SISTEMA DE ENTRENAMIENTO Y PREDICCIÓN - SERIES TEMPORALES")
    print("="*70)
    print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuración: {CONFIG}")
    
    # 1. Cargar datos
    df, feature_cols = load_and_prepare_data()
    
    # 2. Preparar datasets
    X_train, y_train, X_val, y_val, X_test, y_test, mean, std = prepare_datasets(df, feature_cols)
    
    n_features = X_train.shape[-1]
    
    # 4. Entrenar modelos
    tft_model, tft_history = train_tft_model(X_train, y_train, X_val, y_val, n_features)
    lstm_model, lstm_history = train_lstm_model(X_train, y_train, X_val, y_val, n_features)
    
    # 5. Evaluar sobreajuste
    tft_results = evaluate_overfitting(
        tft_model, "TFT", X_train, y_train, X_val, y_val, X_test, y_test
    )
    lstm_results = evaluate_overfitting(
        lstm_model, "LSTM", X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # 6. Fechas alineadas con y y corte para el tramo de Test
    all_target_dates = get_target_dates(df, CONFIG["lookback_steps"], CONFIG["forecast_horizon"])
    train_len = len(y_train)
    val_len = len(y_val) if y_val is not None else 0
    test_len = len(y_test) if y_test is not None else 0
    test_start = train_len + val_len
    test_end = test_start + test_len
    test_dates = all_target_dates.iloc[test_start:test_end]

    # 7. Predicciones futuras a 12 meses: arrancar desde el FINAL del tramo Test
    if test_len > 0:
        cutoff_date = pd.to_datetime(test_dates.iloc[-1])
        df_cutoff = df[df["date"] <= cutoff_date].reset_index(drop=True)
    else:
        df_cutoff = df

    tft_future_preds = predict_future(
        tft_model, "TFT", df_cutoff, feature_cols, mean, std, n_months=CONFIG["future_months"]
    )
    lstm_future_preds = predict_future(
        lstm_model, "LSTM", df_cutoff, feature_cols, mean, std, n_months=CONFIG["future_months"]
    )

    # 8. Guardar resultados
    save_results(tft_results, lstm_results, tft_future_preds, lstm_future_preds)

    # 9. Visualizaciones
    plot_training_history(tft_history, lstm_history)
    plot_results(test_dates, y_test, tft_model, lstm_model, X_test, tft_future_preds, lstm_future_preds)
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO")
    print("="*70)
    
    return {
        "tft_model": tft_model,
        "lstm_model": lstm_model,
        "tft_results": tft_results,
        "lstm_results": lstm_results,
        "tft_future": tft_future_preds,
        "lstm_future": lstm_future_preds,
    }


if __name__ == "__main__":
    results = main()