"""test2.py

Entrena modelo TST (Time Series Transformer) multi-horizonte (predice 12 meses 
en una sola predicción) usando exógenas futuras cuando están disponibles.
Variables soportadas:
- Banrep (Colombia): Inflación, PIB, TRM, Tasa interés, IPP
- Externas: FAO Food Price Index, Brent Oil
Genera:
- Predicción sobre todo el dataset (t+1) para ver ajuste general.
- Predicción sobre conjunto de prueba (t+1) + forecast de 12 meses posteriores en una sola gráfica.

Nota importante:
- Para predecir 12 meses posteriores usando exógenas futuras reales, debes proveer un
  DataFrame con esas exógenas (ver `build_future_exog_df`). Si no se proveen, el script
  repetirá el último valor conocido (equivalente a “congelar” exógenas), pero la
  arquitectura sí soporta exógenas futuras.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Agregar raíz del proyecto al path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.model.model import TFTModel  # solo usamos utilidades de ETL/splits


# ============================
# CONFIG
# ============================
@dataclass
class Config:
    lookback_steps: int = 24
    forecast_horizon: int = 12
    target_col: str = "Inflacion_total"

    # splits
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # training
    epochs: int = 120
    batch_size: int = 16
    patience: int = 15
    learning_rate: float = 1e-3

    # models
    d_model: int = 64
    num_heads: int = 4
    dropout_rate: float = 0.1


CFG = Config()

PROC_DIR = ROOT_DIR / "data" / "proc"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Descripciones de variables para mejor interpretación
VARIABLE_DESCRIPTIONS = {
    "Inflacion_total": "Inflación total (%)",
    "IPP": "Índice Precios Productor",
    "PIB_real_trimestral_2015_AE": "PIB real trimestral",
    "Tasa_interes_colocacion_total": "Tasa interés colocación",
    "TRM": "Tasa cambio COP/USD",
    "Brent": "Precio petróleo Brent",
    "FAO": "Índice precios alimentos FAO",
}


# ============================
# DATA
# ============================
def load_df(proc_dir: Path) -> pd.DataFrame:
    df = TFTModel.load_latest_proc_csv(proc_dir)
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "date"]


def get_future_exog_cols(feature_cols: Iterable[str], target_col: str) -> list[str]:
    # Para el decoder usamos solo exógenas (sin el target)
    return [c for c in feature_cols if c != target_col]


def make_windows_with_future_exog(
    df: pd.DataFrame,
    past_feature_cols: Iterable[str],
    future_exog_cols: Iterable[str],
    target_col: str,
    lookback_steps: int,
    forecast_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crea ventanas para modelos encoder-decoder.

    - X_past: (N, lookback_steps, n_past_features)
    - X_future_exog: (N, forecast_horizon, n_future_exog)
    - y: (N, forecast_horizon)

    Aquí sí incorporamos exógenas futuras (conocidas históricamente) para entrenar.
    """
    past_feature_cols = list(past_feature_cols)
    future_exog_cols = list(future_exog_cols)

    for c in [*past_feature_cols, *future_exog_cols, target_col]:
        if c not in df.columns:
            raise ValueError(f"Columna requerida no está en df: {c}")

    n_total = len(df)
    max_start = n_total - lookback_steps - forecast_horizon + 1
    if max_start <= 0:
        raise ValueError(
            f"No hay suficientes filas ({n_total}) para lookback={lookback_steps} y horizon={forecast_horizon}."
        )

    X_past = []
    X_fut = []
    y = []

    for i in range(max_start):
        past_slice = df.iloc[i : i + lookback_steps]
        fut_slice = df.iloc[i + lookback_steps : i + lookback_steps + forecast_horizon]

        X_past.append(past_slice[past_feature_cols].to_numpy(dtype=np.float32))
        X_fut.append(fut_slice[future_exog_cols].to_numpy(dtype=np.float32))
        y.append(fut_slice[target_col].to_numpy(dtype=np.float32))

    return np.stack(X_past, axis=0), np.stack(X_fut, axis=0), np.stack(y, axis=0)


def standardize_3d_from_train(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Estandariza (B,T,F) usando estadísticas del train (por feature)."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    def transform(X: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if X is None or len(X) == 0:
            return None
        return ((X - mean) / std).astype(np.float32)

    return transform(X_train), transform(X_val), transform(X_test), mean, std


def time_split_3(X: np.ndarray, Xf: np.ndarray, y: np.ndarray, val_ratio: float, test_ratio: float):
    """Split temporal consistente para (X_past, X_fut, y)."""
    n = len(X)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("val_ratio/test_ratio demasiado grandes")

    X_train, y_train, Xf_train = X[:n_train], y[:n_train], Xf[:n_train]
    X_val, y_val, Xf_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val], Xf[n_train : n_train + n_val]
    X_test, y_test, Xf_test = X[n_train + n_val :], y[n_train + n_val :], Xf[n_train + n_val :]

    return (X_train, Xf_train, y_train), (X_val, Xf_val, y_val), (X_test, Xf_test, y_test)


def get_target_dates(df: pd.DataFrame, lookback_steps: int, forecast_horizon: int) -> pd.DatetimeIndex:
    """Fechas base de cada ventana (la fecha del primer paso de forecast)."""
    start = lookback_steps
    end = len(df) - forecast_horizon + 1
    return pd.DatetimeIndex(df["date"].iloc[start:end].reset_index(drop=True))


# ============================
# MODELS
# ============================
def transformer_block(x, num_heads: int, key_dim: int, dropout: float):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = LayerNormalization()(x + attn)
    ff = Dense(key_dim * num_heads, activation="relu")(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(key_dim * num_heads)(ff)
    x = LayerNormalization()(x + ff)
    return x


def build_tst_transformer(
    lookback_steps: int,
    n_past_features: int,
    forecast_horizon: int,
    n_future_exog: int,
    d_model: int,
    num_heads: int,
    dropout_rate: float,
) -> Model:
    """Transformer encoder-decoder simple para series temporales (TST)."""
    past_in = Input(shape=(lookback_steps, n_past_features), name="past_inputs")
    fut_in = Input(shape=(forecast_horizon, n_future_exog), name="future_exog")

    # Proyecciones
    enc = Dense(d_model, name="enc_proj")(past_in)
    dec = Dense(d_model, name="dec_proj")(fut_in)

    # Encoder self-attn
    enc = transformer_block(enc, num_heads=num_heads, key_dim=max(8, d_model // num_heads), dropout=dropout_rate)
    enc = transformer_block(enc, num_heads=num_heads, key_dim=max(8, d_model // num_heads), dropout=dropout_rate)

    # Decoder: self-attn + cross-attn
    dec_self = MultiHeadAttention(num_heads=num_heads, key_dim=max(8, d_model // num_heads), dropout=dropout_rate)(dec, dec)
    dec = LayerNormalization()(dec + dec_self)

    cross = MultiHeadAttention(num_heads=num_heads, key_dim=max(8, d_model // num_heads), dropout=dropout_rate)(dec, enc)
    dec = LayerNormalization()(dec + cross)

    ff = Dense(d_model * 2, activation="relu")(dec)
    ff = Dropout(dropout_rate)(ff)
    ff = Dense(d_model)(ff)
    dec = LayerNormalization()(dec + ff)

    yhat = TimeDistributed(Dense(1), name="yhat")(dec)

    model = Model(inputs=[past_in, fut_in], outputs=yhat, name="TST_Transformer")
    return model


# ============================
# METRICS / PRED
# ============================
def flatten_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mae = float(np.mean(np.abs(yt - yp)))
    mse = float(np.mean((yt - yp) ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((yt - yp) / (yt + 1e-8))) * 100)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


def predict_tplus1_from_direct12(y_pred_h: np.ndarray) -> np.ndarray:
    """Convierte (N, horizon, 1) -> (N,) usando el primer paso (t+1)."""
    return y_pred_h[:, 0, 0]


def build_future_exog_df(
    df: pd.DataFrame,
    future_exog_cols: list[str],
    horizon: int,
) -> pd.DataFrame:
    """Construye un DataFrame de exógenas futuras para el forecast.

    Por defecto REPITE el último valor conocido para cada exógena.
    Si quieres escenarios reales, reemplaza esta función para llenar valores futuros.
    """
    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")

    last_row = df.iloc[-1]
    data = {"date": future_dates}
    for c in future_exog_cols:
        data[c] = [float(last_row[c])] * horizon

    return pd.DataFrame(data)


def make_last_inputs_for_forecast(
    df: pd.DataFrame,
    past_feature_cols: list[str],
    future_exog_df: pd.DataFrame,
    future_exog_cols: list[str],
    lookback_steps: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    past = df[past_feature_cols].iloc[-lookback_steps:].to_numpy(dtype=np.float32)[None, :, :]
    fut = future_exog_df[future_exog_cols].iloc[:horizon].to_numpy(dtype=np.float32)[None, :, :]
    future_dates = pd.DatetimeIndex(future_exog_df["date"].iloc[:horizon])
    return past, fut, future_dates


# ============================
# PLOTS
# ============================
def plot_full_dataset_tplus1(
    df: pd.DataFrame,
    base_dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred_t1: np.ndarray,
    title: str,
    out_path: Path,
):
    plt.figure(figsize=(16, 6))
    # y_true es (N,h) -> t+1 en y_true[:,0]
    plt.plot(base_dates, y_true[:, 0], label="Real (t+1)", color="blue", alpha=0.7)
    plt.plot(base_dates, y_pred_t1, label="Pred (t+1)", color="darkorange", alpha=0.8)
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel(CFG.target_col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_test_and_future(
    test_dates: pd.DatetimeIndex,
    y_test: np.ndarray,
    y_test_pred_t1: np.ndarray,
    tail_dates: Optional[pd.DatetimeIndex],
    tail_true: Optional[np.ndarray],
    tail_pred: Optional[np.ndarray],
    future_dates: pd.DatetimeIndex,
    future_pred: np.ndarray,
    title: str,
    out_path: Path,
):
    plt.figure(figsize=(16, 6))
    plt.plot(test_dates, y_test[:, 0], label="Test Real (t+1)", color="blue", alpha=0.7)
    plt.plot(test_dates, y_test_pred_t1, label="Test Pred (t+1)", color="green", alpha=0.85)

    # Coser el tramo final: usar el último window (t+2..t+12) para cubrir hasta el final del dataset
    if tail_dates is not None and tail_true is not None and tail_pred is not None:
        plt.plot(tail_dates, tail_true, label="Real (cola del test)", color="blue", alpha=0.6)
        plt.plot(tail_dates, tail_pred, label="Pred (cola del test)", color="green", alpha=0.75)

    plt.plot(future_dates, future_pred, label="Forecast +12m", color="crimson", linestyle="--")
    plt.axvline(x=test_dates[-1], color="gray", linestyle=":", label="Fin Test / Inicio Forecast")
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel(CFG.target_col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


# ============================
# MAIN
# ============================
def train_and_eval(model: Model, name: str, train, val, test, model_path: Path):
    (Xtr, Xftr, ytr) = train
    (Xva, Xfva, yva) = val
    (Xte, Xfte, yte) = test

    model.compile(optimizer=Adam(CFG.learning_rate), loss="mse")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=CFG.patience, restore_best_weights=True) if Xva is not None and len(Xva) > 0 else EarlyStopping(monitor="loss", patience=CFG.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss" if Xva is not None and len(Xva) > 0 else "loss", factor=0.5, patience=max(2, CFG.patience // 3), min_lr=1e-5),
        ModelCheckpoint(str(model_path), monitor="val_loss" if Xva is not None and len(Xva) > 0 else "loss", save_best_only=True),
    ]

    history = model.fit(
        [Xtr, Xftr], ytr[..., None],
        validation_data=([Xva, Xfva], yva[..., None]) if Xva is not None and len(Xva) > 0 else None,
        epochs=CFG.epochs,
        batch_size=CFG.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    # preds
    pred_tr = model.predict([Xtr, Xftr], verbose=0)
    pred_te = model.predict([Xte, Xfte], verbose=0) if Xte is not None and len(Xte) > 0 else None

    metrics = {
        "Train": flatten_metrics(ytr, pred_tr[:, :, 0]),
    }
    if Xva is not None and len(Xva) > 0:
        pred_va = model.predict([Xva, Xfva], verbose=0)
        metrics["Validation"] = flatten_metrics(yva, pred_va[:, :, 0])
    if pred_te is not None:
        metrics["Test"] = flatten_metrics(yte, pred_te[:, :, 0])

    print(f"\n===== {name} METRICS =====")
    for split, m in metrics.items():
        print(split)
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")

    return history, metrics


def main():
    print("=" * 70)
    print("SISTEMA MULTI-HORIZONTE (12 meses en una sola predicción)")
    print("=" * 70)
    print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {CFG}")

    df = load_df(PROC_DIR)
    feature_cols = get_feature_cols(df)
    future_exog_cols = get_future_exog_cols(feature_cols, CFG.target_col)

    print(f"Filas: {len(df)}")
    print(f"Fechas: {df['date'].min()} → {df['date'].max()}")
    print(f"Target: {CFG.target_col}")
    print(f"\nFeatures del encoder (past) [{len(feature_cols)}]:")
    for col in feature_cols:
        desc = VARIABLE_DESCRIPTIONS.get(col, col)
        print(f"   • {col}: {desc}")
    print(f"\nFeatures del decoder (future exog) [{len(future_exog_cols)}]:")
    for col in future_exog_cols:
        desc = VARIABLE_DESCRIPTIONS.get(col, col)
        print(f"   • {col}: {desc}")

    X_past, X_fut, y = make_windows_with_future_exog(
        df=df,
        past_feature_cols=feature_cols,
        future_exog_cols=future_exog_cols,
        target_col=CFG.target_col,
        lookback_steps=CFG.lookback_steps,
        forecast_horizon=CFG.forecast_horizon,
    )

    (train, val, test) = time_split_3(X_past, X_fut, y, CFG.val_ratio, CFG.test_ratio)
    (Xtr, Xftr, ytr) = train
    (Xva, Xfva, yva) = val
    (Xte, Xfte, yte) = test

    # standardize X_past and X_fut separately (train stats)
    Xtr_s, Xva_s, Xte_s, mean_p, std_p = standardize_3d_from_train(Xtr, Xva if len(Xva) > 0 else None, Xte if len(Xte) > 0 else None)
    Xftr_s, Xfva_s, Xfte_s, mean_f, std_f = standardize_3d_from_train(Xftr, Xfva if len(Xfva) > 0 else None, Xfte if len(Xfte) > 0 else None)

    train_s = (Xtr_s, Xftr_s, ytr)
    val_s = (Xva_s, Xfva_s, yva)
    test_s = (Xte_s, Xfte_s, yte)

    print("\nShapes:")
    print(f"  X_train_past: {Xtr_s.shape}, X_train_fut: {Xftr_s.shape}, y_train: {ytr.shape}")
    if Xva_s is not None:
        print(f"  X_val_past: {Xva_s.shape}, X_val_fut: {Xfva_s.shape}, y_val: {yva.shape}")
    if Xte_s is not None:
        print(f"  X_test_past: {Xte_s.shape}, X_test_fut: {Xfte_s.shape}, y_test: {yte.shape}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) TST Transformer
    tst = build_tst_transformer(
        lookback_steps=CFG.lookback_steps,
        n_past_features=Xtr_s.shape[-1],
        forecast_horizon=CFG.forecast_horizon,
        n_future_exog=Xftr_s.shape[-1],
        d_model=CFG.d_model,
        num_heads=CFG.num_heads,
        dropout_rate=CFG.dropout_rate,
    )

    train_and_eval(tst, "TST", train_s, val_s, test_s, MODELS_DIR / "tst_direct12.keras")

    # =========
    # PREDICCIONES PARA GRAFICAR
    # =========
    base_dates = get_target_dates(df, CFG.lookback_steps, CFG.forecast_horizon)

    # Predicción sobre TODO el dataset: usamos el primer paso (t+1) en cada ventana
    X_past_s, _, _, _, _ = standardize_3d_from_train(X_past, None, None)
    X_fut_s, _, _, _, _ = standardize_3d_from_train(X_fut, None, None)

    # OJO: para consistencia, usamos las stats calculadas de train
    X_past_s = ((X_past - mean_p) / std_p).astype(np.float32)
    X_fut_s = ((X_fut - mean_f) / std_f).astype(np.float32)

    tst_pred_all = tst.predict([X_past_s, X_fut_s], verbose=0)[:, :, 0]

    tst_t1 = predict_tplus1_from_direct12(tst_pred_all[..., None])

    out_full_tst = RESULTS_DIR / f"test2_full_tst_{ts}.png"
    plot_full_dataset_tplus1(
        df=df,
        base_dates=base_dates,
        y_true=y,
        y_pred_t1=tst_t1,
        title=f"TST (direct 12) - Predicción sobre todo el dataset (t+1) - {CFG.target_col}",
        out_path=out_full_tst,
    )

    # Test + 12 meses posteriores en una sola gráfica
    train_len = len(ytr)
    val_len = len(yva) if yva is not None else 0
    test_len = len(yte) if yte is not None else 0

    test_start = train_len + val_len
    test_end = test_start + test_len
    test_dates = base_dates[test_start:test_end]

    if test_len > 0:
        Xte_p = ((Xte - mean_p) / std_p).astype(np.float32)
        Xte_f = ((Xfte - mean_f) / std_f).astype(np.float32)

        tst_pred_test = tst.predict([Xte_p, Xte_f], verbose=0)[:, :, 0]

        tst_test_t1 = tst_pred_test[:, 0]

        # Cola para evitar el hueco de 12 meses: usar el último window del test
        last_base = pd.Timestamp(test_dates[-1])
        tail_dates = pd.date_range(start=last_base + pd.DateOffset(months=1), periods=CFG.forecast_horizon - 1, freq="MS")
        tail_true = yte[-1, 1:]
        tail_pred_tst = tst_pred_test[-1, 1:]

        # Forecast +12m desde el FINAL de todo el df
        future_exog_df = build_future_exog_df(df, future_exog_cols, CFG.forecast_horizon)
        past_last, fut_last, future_dates = make_last_inputs_for_forecast(
            df, feature_cols, future_exog_df, future_exog_cols, CFG.lookback_steps, CFG.forecast_horizon
        )

        past_last_s = ((past_last - mean_p) / std_p).astype(np.float32)
        fut_last_s = ((fut_last - mean_f) / std_f).astype(np.float32)

        tst_future = tst.predict([past_last_s, fut_last_s], verbose=0)[0, :, 0]

        out_test_future_tst = RESULTS_DIR / f"test2_test_plus_future_tst_{ts}.png"
        plot_test_and_future(
            test_dates=test_dates,
            y_test=yte,
            y_test_pred_t1=tst_test_t1,
            tail_dates=pd.DatetimeIndex(tail_dates),
            tail_true=tail_true,
            tail_pred=tail_pred_tst,
            future_dates=future_dates,
            future_pred=tst_future,
            title=f"TST (direct 12) - Test (t+1) + Forecast +12m - {CFG.target_col}",
            out_path=out_test_future_tst,
        )
    else:
        print("No hay test set (test_ratio=0).")

    print("\nOK. Figuras guardadas en results/ con prefijo test2_")


if __name__ == "__main__":
    main()
