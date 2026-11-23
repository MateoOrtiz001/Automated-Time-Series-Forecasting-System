"""
Dashboard de Predicción de Inflación - Colombia
================================================

Dashboard interactivo en Streamlit para visualizar:
- Predicciones de inflación a 6 meses (modelo TFT)
- Datos históricos de variables macroeconómicas
- Covariables futuras pronosticadas (IPP, TRM, Brent, FAO, Tasa, PIB)
- Métricas del modelo
- Estado del sistema

Uso:
    streamlit run src/webApp/app.py
"""

import sys
from pathlib import Path

# Configurar paths
APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

TFTModel = None
add_calendar_features = None
forecast_all_covariates = None
get_scaler_stats = None
IMPORT_ERROR = None

try:
    from src.model.model import TFTModel
except Exception as exc:
    IMPORT_ERROR = f"No se pudo cargar el modelo TFT: {exc}"

try:
    from src.pipeline.core import (
        add_calendar_features,
        forecast_all_covariates,
        get_forecast_dates,
        get_scaler_stats,
    )
except Exception as exc:
    core_error = f"No se pudo cargar el pipeline: {exc}"
    IMPORT_ERROR = core_error if IMPORT_ERROR is None else f"{IMPORT_ERROR} | {core_error}"


def get_forecast_dates(n_months, reference_date=None):
    """Crea el horizonte futuro sin depender del último dato histórico."""
    reference_date = reference_date or datetime.now()
    current_month = pd.Timestamp(reference_date).to_period("M").to_timestamp(how="start")
    first_month = current_month + pd.offsets.MonthBegin(1)
    return pd.date_range(start=first_month, periods=n_months, freq="MS")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Predicción de Inflación - Colombia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Rutas
DATA_PROC_DIR = ROOT_DIR / "data" / "proc"
MODELS_DIR = ROOT_DIR / "models"
MISC_DIR = ROOT_DIR / "misc"
MISC_MODELS_DIR = MISC_DIR / "models"
MISC_RESULTS_DIR = MISC_DIR / "results"
PIPELINE_STATE_FILE = ROOT_DIR / "src" / "pipeline" / "pipeline_state.json"
LEGACY_PIPELINE_STATE_FILE = MISC_DIR / "pipeline_state.json"

# Configuración del modelo
CONFIG = {
    "lookback_steps": 12,
    "forecast_horizon": 6,
    "target_col": "Inflacion_total",
    "future_months": 6,
    "tft_units": 48,
    "tft_heads": 2,
    "tft_lstm_layers": 1,
    "tft_grn_layers": 1,
    "tft_dropout": 0.1,
    # Covariables futuras
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
    "future_feature_cols": [
        "sin_month", "cos_month", "IPP",
        "TRM", "Brent", "FAO",
        "Tasa_interes_colocacion_total", "PIB_real_trimestral_2015_AE",
    ],
    "covariate_forecast_steps": 6,
}

if add_calendar_features is None:
    def add_calendar_features(df):
        df = df.copy()
        month = df["date"].dt.month
        df["sin_month"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
        df["cos_month"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
        return df


if forecast_all_covariates is None:
    def forecast_all_covariates(df, n_steps=None, logger=None):
        n_steps = n_steps or CONFIG["future_months"]
        forecasts = {}
        for col in CONFIG["future_forecast_cols"]:
            if col in df.columns:
                forecasts[col] = np.full(n_steps, float(df[col].iloc[-1]), dtype=np.float32)
        return forecasts


# Descripciones de variables
VARIABLE_INFO = {
    "Inflacion_total": {
        "name": "Inflación Total",
        "description": "Variación anual del IPC (%)",
        "unit": "%",
        "color": "#E74C3C",
    },
    "IPP": {
        "name": "Índice de Precios del Productor",
        "description": "Variación del IPP",
        "unit": "índice",
        "color": "#3498DB",
    },
    "PIB_real_trimestral_2015_AE": {
        "name": "PIB Real Trimestral",
        "description": "PIB real con año base 2015",
        "unit": "billones COP",
        "color": "#2ECC71",
    },
    "Tasa_interes_colocacion_total": {
        "name": "Tasa de Interés",
        "description": "Tasa de interés de colocación",
        "unit": "%",
        "color": "#9B59B6",
    },
    "TRM": {
        "name": "Tasa de Cambio (TRM)",
        "description": "Tasa representativa del mercado COP/USD",
        "unit": "COP/USD",
        "color": "#F39C12",
    },
    "Brent": {
        "name": "Petróleo Brent",
        "description": "Precio del petróleo Brent",
        "unit": "USD/barril",
        "color": "#1ABC9C",
    },
    "FAO": {
        "name": "Índice FAO",
        "description": "Índice de precios de alimentos FAO",
        "unit": "índice",
        "color": "#E67E22",
    },
}


# =============================================================================
# FUNCIONES DE CARGA DE DATOS
# =============================================================================
def get_latest_data_hash():
    """Obtiene un hash basado en la fecha máxima de los datos disponibles.
    
    Esto permite invalidar el caché de Streamlit cuando hay datos más recientes.
    """
    try:
        df = load_data_internal()
        if df is not None:
            max_date = df["date"].max()
            return f"{max_date.strftime('%Y%m%d')}_{len(df)}"
    except Exception:
        pass
    return "default"


def load_data_internal():
    """Carga datos priorizando latest.csv (para Streamlit Cloud)."""
    # Priorizar latest.csv (archivo estático para Streamlit Cloud)
    latest_file = DATA_PROC_DIR / "latest.csv"
    if latest_file.exists():
        df = pd.read_csv(latest_file)
        df["date"] = pd.to_datetime(df["date"])
        return df.dropna().reset_index(drop=True)
    
    # Fallback: buscar el CSV más reciente por timestamp
    if TFTModel is not None:
        return TFTModel.load_latest_proc_csv(DATA_PROC_DIR)
    return None


@st.cache_data(ttl=300)  # 5 minutos de TTL
def load_data(_cache_key: str = None):
    """Carga los datos procesados más recientes.
    
    El parámetro _cache_key (con _ para que Streamlit lo ignore en el display)
    permite invalidar el caché cuando hay datos nuevos.
    """
    try:
        df = load_data_internal()
        if df is not None:
            return df.dropna().reset_index(drop=True)
        return None
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None


def get_latest_model_name():
    """Obtiene el nombre del modelo más reciente para control de caché."""
    try:
        if MISC_MODELS_DIR.exists():
            finetuned = sorted(MISC_MODELS_DIR.glob("tft_finetuned_*.keras"), reverse=True)
            if finetuned:
                return finetuned[0].name
        return "tft_base.keras"
    except Exception:
        return "default"


def create_tft_instance_for_inference(feature_cols, future_feature_cols):
    """Crea instancia de TFTModel compatible con distintas firmas del constructor.

    Streamlit Cloud puede quedar temporalmente con artefactos o dependencias
    desalineadas; este fallback evita fallar por kwargs no soportados.
    """
    common_kwargs = dict(
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
        n_features=len(feature_cols),
        units=CONFIG["tft_units"],
        num_heads=CONFIG["tft_heads"],
        num_grn_layers=CONFIG["tft_grn_layers"],
        dropout_rate=CONFIG["tft_dropout"],
        num_quantiles=3,
    )

    # Firma completa actual
    try:
        return TFTModel(
            **common_kwargs,
            n_future_features=len(future_feature_cols),
            num_lstm_layers=CONFIG["tft_lstm_layers"],
        )
    except TypeError:
        pass

    # Firma intermedia (sin num_lstm_layers)
    try:
        return TFTModel(
            **common_kwargs,
            n_future_features=len(future_feature_cols),
        )
    except TypeError:
        pass

    # Firma mínima (retrocompatibilidad)
    return TFTModel(**common_kwargs)


@st.cache_resource
def load_model(_model_key: str = None):
    """Carga el modelo TFT más reciente.
    
    El parámetro _model_key permite invalidar el caché cuando hay un modelo nuevo.
    """
    candidate_paths = []
    if MISC_MODELS_DIR.exists():
        finetuned = sorted(MISC_MODELS_DIR.glob("tft_finetuned_*.keras"), reverse=True)
        candidate_paths.extend(finetuned)

    candidate_paths.extend([
        MISC_MODELS_DIR / "tft_base.keras",
        MODELS_DIR / "tft_best.keras",
    ])

    unique_candidates = []
    for p in candidate_paths:
        if p.exists() and p not in unique_candidates:
            unique_candidates.append(p)

    if not unique_candidates:
        return None, None
    
    # Cargar datos para obtener n_features (incluye calendar features)
    try:
        df = TFTModel.load_latest_proc_csv(DATA_PROC_DIR)
        df = df.dropna().reset_index(drop=True)
        df = add_calendar_features(df)
    except Exception as e:
        st.error(f"Error cargando datos para modelo: {e}")
        return None, None
    
    feature_cols = [c for c in df.columns if c != "date"]
    future_feature_cols = CONFIG.get("future_feature_cols", [])
    
    tft = create_tft_instance_for_inference(feature_cols, future_feature_cols)
    
    load_errors = []
    for model_path in unique_candidates:
        try:
            tft.build_model()
            tft.model.load_weights(str(model_path))
            return tft, model_path.name
        except Exception as e:
            load_errors.append(f"{model_path.name}: {e}")

    st.error("No se pudo cargar ningún modelo compatible para inferencia.")
    st.caption(" | ".join(load_errors))
    return None, None


def load_pipeline_state():
    """Carga el estado del pipeline."""
    for state_file in [PIPELINE_STATE_FILE, LEGACY_PIPELINE_STATE_FILE]:
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def load_latest_predictions():
    """Carga las predicciones más recientes si siguen el horizonte actual."""
    if not MISC_RESULTS_DIR.exists():
        return None

    latest_file = MISC_RESULTS_DIR / "predictions_latest.csv"
    if latest_file.exists():
        df = pd.read_csv(latest_file, parse_dates=["date"])
        expected_dates = get_forecast_dates(CONFIG["future_months"])
        stored_dates = pd.DatetimeIndex(df["date"])
        if not stored_dates.equals(expected_dates):
            return None
        return df
    


def generate_predictions(model, df, n_months=None, covariate_forecasts=None):
    """Genera predicciones directas con decoder de covariables futuras.
    
    El modelo recibe dos entradas:
    - past_inputs: ventana de lookback estandarizada.
    - future_inputs: covariables futuras estandarizadas.
    
    Las estadísticas de estandarización se cargan desde disco (guardadas
    durante el entrenamiento) para garantizar consistencia.
    """
    n_months = n_months or CONFIG["future_months"]
    future_feature_cols = CONFIG.get("future_feature_cols", [])
    
    # Agregar calendar features si no están
    if "sin_month" not in df.columns:
        df = add_calendar_features(df)
    
    feature_cols = [c for c in df.columns if c != "date"]
    
    # Cargar estadísticas de estandarización guardadas
    if get_scaler_stats is None:
        data_features = df[feature_cols].to_numpy(dtype=np.float32)
        mean = data_features.mean(axis=0)
        std = np.where(data_features.std(axis=0) < 1e-6, 1.0, data_features.std(axis=0))
        mean_f = std_f = mean_y = std_y = None
    else:
        mean, std, mean_f, std_f, mean_y, std_y = get_scaler_stats(df, feature_cols, logger=None)
    
    # Preparar entrada del encoder (lookback)
    data_features = df[feature_cols].to_numpy(dtype=np.float32)
    current_window = data_features[-CONFIG["lookback_steps"]:].copy()
    current_window_std = ((current_window - mean) / std).astype(np.float32)
    X_past = current_window_std[np.newaxis, :, :]  # (1, lookback, n_features)
    
    # Preparar entrada del decoder (features futuras)
    future_dates = get_forecast_dates(n_months)
    
    # Pronosticar todas las covariables futuras
    covariate_forecasts = covariate_forecasts or forecast_all_covariates(df, n_steps=n_months)
    
    # Construir features futuras
    X_fut = np.zeros((1, n_months, len(future_feature_cols)), dtype=np.float32)
    
    for i, date in enumerate(future_dates):
        month = date.month
        for j, col in enumerate(future_feature_cols):
            if col == "sin_month":
                X_fut[0, i, j] = np.sin(2 * np.pi * month / 12)
            elif col == "cos_month":
                X_fut[0, i, j] = np.cos(2 * np.pi * month / 12)
            elif col in covariate_forecasts and i < len(covariate_forecasts[col]):
                X_fut[0, i, j] = covariate_forecasts[col][i]
            else:
                X_fut[0, i, j] = float(df[col].iloc[-1])
    
    # Estandarizar features futuras
    if mean_f is not None and std_f is not None:
        X_fut = ((X_fut - mean_f) / std_f).astype(np.float32)
    
    # Predicción directa (una sola pasada)
    pred_result = model.predict([X_past, X_fut])
    
    # Extraer predicciones: (1, horizon, quantiles)
    median = np.ravel(pred_result.get("median", pred_result["predictions"][..., 1]))
    lower = np.ravel(pred_result["lower"]) if "lower" in pred_result else [None] * n_months
    upper = np.ravel(pred_result["upper"]) if "upper" in pred_result else [None] * n_months
    
    # Desnormalizar predicciones a escala original
    if mean_y is not None and std_y is not None:
        median = median * std_y + mean_y
        if not isinstance(lower, list):
            lower = lower * std_y + mean_y
        if not isinstance(upper, list):
            upper = upper * std_y + mean_y
    
    # Corregir cuantiles invertidos: banda IC usa extremos de los 3 cuantiles,
    # manteniendo la predicción (q50) siempre dentro del intervalo.
    if not isinstance(lower, list) and not isinstance(upper, list):
        all_q = np.stack([lower, median, upper], axis=-1)  # (horizon, 3)
        lower = np.min(all_q, axis=-1)
        upper = np.max(all_q, axis=-1)
    
    predictions = []
    for i in range(n_months):
        predictions.append({
            "date": future_dates[i],
            "prediction": float(median[i]),
            "lower": float(lower[i]) if lower[i] is not None else None,
            "upper": float(upper[i]) if upper[i] is not None else None,
        })
    
    return pd.DataFrame(predictions)


def generate_future_covariates_df(df, n_months=None, forecasts=None):
    """Genera DataFrame con covariables futuras pronosticadas para visualización."""
    n_months = n_months or CONFIG["future_months"]

    # Asegurar calendar features para coherencia con el pipeline
    if "sin_month" not in df.columns:
        df = add_calendar_features(df)

    future_dates = get_forecast_dates(n_months)

    forecasts = forecasts or forecast_all_covariates(df, n_steps=n_months)
    covariate_cols = list(CONFIG.get("future_forecast_cols", {}).keys())

    rows = []
    for i in range(n_months):
        row = {"date": future_dates[i]}
        for col in covariate_cols:
            if col in forecasts and i < len(forecasts[col]):
                row[col] = float(forecasts[col][i])
            else:
                row[col] = float(df[col].iloc[-1]) if col in df.columns else None
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# COMPONENTES DE VISUALIZACIÓN
# =============================================================================
def plot_inflation_forecast(df_hist, df_pred, months_history=36):
    """Gráfico principal de predicción de inflación."""
    fig = go.Figure()
    
    # Datos históricos
    recent = df_hist.tail(months_history)
    fig.add_trace(go.Scatter(
        x=recent["date"],
        y=recent["Inflacion_total"],
        mode="lines+markers",
        name="Histórico",
        line=dict(color="#3498DB", width=2),
        marker=dict(size=4),
    ))
    
    # Predicciones
    fig.add_trace(go.Scatter(
        x=df_pred["date"],
        y=df_pred["prediction"],
        mode="lines+markers",
        name="Predicción TFT",
        line=dict(color="#E74C3C", width=3, dash="dash"),
        marker=dict(size=6),
    ))
    
    # Intervalo de confianza q10–q90
    _CI_GROUP = "IC q10–q90"
    if "lower" in df_pred.columns and df_pred["lower"].notna().any():
        # Lower bound line (dotted, visible, acts as fill anchor)
        fig.add_trace(go.Scatter(
            x=df_pred["date"],
            y=df_pred["lower"],
            mode="lines",
            line=dict(color="rgba(231, 76, 60, 0.5)", width=1, dash="dot"),
            legendgroup=_CI_GROUP,
            name=_CI_GROUP,
            showlegend=True,
            hovertemplate="q10: %{y:.2f}%<extra></extra>",
        ))
        # Upper bound line with fill down to lower
        fig.add_trace(go.Scatter(
            x=df_pred["date"],
            y=df_pred["upper"],
            mode="lines",
            line=dict(color="rgba(231, 76, 60, 0.5)", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(231, 76, 60, 0.15)",
            legendgroup=_CI_GROUP,
            showlegend=False,
            hovertemplate="q90: %{y:.2f}%<extra></extra>",
        ))
    
    # Línea vertical de corte
    # Nota: en algunas versiones de Plotly, `annotation_text` en `add_vline` falla
    # con ejes de fecha (intenta promediar valores datetime). Por eso, agregamos
    # la anotación por separado.
    last_date = df_hist["date"].iloc[-1]
    if hasattr(last_date, "to_pydatetime"):
        last_date = last_date.to_pydatetime()

    fig.add_vline(
        x=last_date,
        line_dash="dot",
        line_color="gray",
    )

    fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="Último dato real",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="gray"),
    )
    
    # Meta de inflación del Banco de la República
    fig.add_hline(y=3, line_dash="dash", line_color="green", 
                  annotation_text="Meta BanRep (3%)")
    fig.add_hrect(y0=2, y1=4, fillcolor="green", opacity=0.1,
                  annotation_text="Rango meta")
    
    fig.update_layout(
        title="Predicción de Inflación - Colombia",
        xaxis_title="Fecha",
        yaxis_title="Inflación (%)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
    )
    
    return fig


def plot_variable_history(df, variable, months=60):
    """Gráfico de historia de una variable."""
    info = VARIABLE_INFO.get(variable, {"name": variable, "color": "#333", "unit": ""})
    recent = df.tail(months)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["date"],
        y=recent[variable],
        mode="lines",
        name=info["name"],
        line=dict(color=info["color"], width=2),
        fill="tozeroy",
        fillcolor=f"rgba{tuple(list(int(info['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}",
    ))
    
    fig.update_layout(
        title=f"{info['name']} - Últimos {months} meses",
        xaxis_title="Fecha",
        yaxis_title=info["unit"],
        hovermode="x unified",
        height=350,
    )
    
    return fig


def plot_correlation_matrix(df):
    """Matriz de correlación de variables."""
    numeric_cols = [c for c in df.columns if c != "date"]
    corr = df[numeric_cols].corr()
    
    # Renombrar columnas para mejor visualización
    labels = [VARIABLE_INFO.get(c, {"name": c})["name"][:15] for c in numeric_cols]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title="Matriz de Correlación",
        height=450,
    )
    
    return fig


def plot_multi_variable(df, variables, months=36):
    """Gráfico de múltiples variables normalizadas."""
    recent = df.tail(months).copy()
    
    fig = go.Figure()
    
    for var in variables:
        info = VARIABLE_INFO.get(var, {"name": var, "color": "#333"})
        # Normalizar para comparación
        normalized = (recent[var] - recent[var].mean()) / recent[var].std()
        
        fig.add_trace(go.Scatter(
            x=recent["date"],
            y=normalized,
            mode="lines",
            name=info["name"],
            line=dict(color=info["color"], width=2),
        ))
    
    fig.update_layout(
        title="Comparación de Variables (Normalizadas)",
        xaxis_title="Fecha",
        yaxis_title="Desviaciones estándar",
        hovermode="x unified",
        height=400,
    )
    
    return fig


def plot_future_covariates(df_cov):
    """Gráfico de líneas para covariables futuras pronosticadas."""
    value_cols = [c for c in df_cov.columns if c != "date"]
    long_df = df_cov.melt(id_vars="date", value_vars=value_cols, var_name="Covariable", value_name="Valor")

    fig = px.line(
        long_df,
        x="date",
        y="Valor",
        color="Covariable",
        markers=True,
        title="Covariables Futuras Pronosticadas",
    )
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Valor",
        hovermode="x unified",
        height=420,
    )
    return fig


def plot_future_covariates_selected(df_cov, selected_covariates):
    """Gráfico de líneas para covariables futuras filtradas por selección."""
    if not selected_covariates:
        return None

    value_cols = [c for c in selected_covariates if c in df_cov.columns and c != "date"]
    if not value_cols:
        return None

    long_df = df_cov.melt(id_vars="date", value_vars=value_cols, var_name="Covariable", value_name="Valor")

    fig = px.line(
        long_df,
        x="date",
        y="Valor",
        color="Covariable",
        markers=True,
        title="Covariables Futuras Pronosticadas",
    )
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Valor",
        hovermode="x unified",
        height=420,
    )
    return fig


# =============================================================================
# APLICACIÓN PRINCIPAL
# =============================================================================
def main():
    if IMPORT_ERROR:
        st.warning("Se cargó el dashboard sin todos los componentes de predicción.")
        with st.expander("Detalle técnico"):
            st.code(IMPORT_ERROR)

    # Header
    st.title("Sistema de Predicción de Inflación")
    st.markdown("**Colombia - Modelo TFT (Temporal Fusion Transformer)**")
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Flag_of_Colombia.svg/320px-Flag_of_Colombia.svg.png", width=150)
        st.header("Configuración")
        
        # Cargar estado del pipeline
        pipeline_state = load_pipeline_state()
        
        st.subheader("Estado del Sistema")
        if pipeline_state:
            last_pred = pipeline_state.get("last_prediction", "N/A")
            if last_pred != "N/A":
                last_pred = datetime.fromisoformat(last_pred).strftime("%Y-%m-%d %H:%M")
            st.info(f"Última predicción: {last_pred}")
            
            last_ft = pipeline_state.get("last_finetune", "N/A")
            if last_ft != "N/A":
                last_ft = datetime.fromisoformat(last_ft).strftime("%Y-%m-%d")
            st.info(f"Último fine-tuning: {last_ft}")
            
            model_name = pipeline_state.get("current_model", "tft_base.keras")
            st.info(f"Modelo: {model_name}")
        
        st.divider()
        
        # Controles de visualización
        st.subheader("Visualización")
        months_history = st.slider(
            "Meses de historial",
            min_value=12,
            max_value=120,
            value=36,
            step=6,
        )
        
        show_confidence = st.checkbox("Mostrar intervalo de confianza", value=True)
        
        st.divider()
        
        # Botón para actualizar datos (limpiar caché)
        st.subheader("🔄 Actualización")
        if st.button("Refrescar datos", help="Limpia el caché y recarga los datos más recientes"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Cargar datos con caché inteligente (se invalida cuando hay datos más recientes)
    data_cache_key = get_latest_data_hash()
    df = load_data(_cache_key=data_cache_key)
    if df is None:
        st.error("No se pudieron cargar los datos. Verifica que existan archivos en data/proc/")
        return
    
    # Mostrar información del rango de datos en la barra lateral
    with st.sidebar:
        st.subheader("📊 Datos Cargados")
        data_start = df["date"].min().strftime("%Y-%m")
        data_end = df["date"].max().strftime("%Y-%m")
        st.success(f"Rango: {data_start} → {data_end}")
        st.info(f"Total: {len(df)} meses")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "Predicciones", 
        "Datos Históricos", 
        "Análisis",
        "Información"
    ])
    
    # ==========================================================================
    # TAB 1: PREDICCIONES
    # ==========================================================================
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Predicción de Inflación a 6 Meses")
            
            # Cargar o generar predicciones (con caché que se invalida cuando hay modelo nuevo)
            model_cache_key = f"{get_latest_model_name()}_{data_cache_key}"
            model, model_name = (None, None) if TFTModel is None else load_model(_model_key=model_cache_key)

            covariate_forecasts = None
            if model is not None:
                covariate_forecasts = forecast_all_covariates(
                    df,
                    n_steps=CONFIG["future_months"],
                )
            
            if model is None:
                st.warning("No se encontró modelo entrenado. Entrena el modelo primero.")
                pred_df = load_latest_predictions()
            elif st.session_state.get("regenerate", False):
                with st.spinner("Generando predicciones..."):
                    pred_df = generate_predictions(model, df, covariate_forecasts=covariate_forecasts)
                    st.session_state["regenerate"] = False
                    st.success(" Predicciones generadas")
            else:
                # Intentar cargar predicciones existentes o generar nuevas
                pred_df = load_latest_predictions()
                if pred_df is None and model is not None:
                    with st.spinner("Generando predicciones..."):
                        pred_df = generate_predictions(model, df, covariate_forecasts=covariate_forecasts)
            
            if pred_df is not None:
                fig = plot_inflation_forecast(df, pred_df, months_history)
                if not show_confidence:
                    fig.data = [t for t in fig.data if t.legendgroup != "IC q10–q90"]
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("No hay predicciones disponibles")
        
        with col2:
            st.subheader(" Resumen")
            
            if pred_df is not None:
                # Último valor real
                last_real = df["Inflacion_total"].iloc[-1]
                last_date = df["date"].iloc[-1].strftime("%Y-%m")
                
                st.metric(
                    label=f"Último dato real ({last_date})",
                    value=f"{last_real:.2f}%",
                )
                
                st.divider()
                
                # Predicciones destacadas
                st.markdown("**Predicciones:**")
                for i, row in pred_df.head(3).iterrows():
                    delta = row["prediction"] - last_real
                    st.metric(
                        label=row["date"].strftime("%Y-%m"),
                        value=f"{row['prediction']:.2f}%",
                        delta=f"{delta:+.2f}%",
                        delta_color="inverse",
                    )
                
                st.divider()
                
                # Promedio anual predicho
                avg_pred = pred_df["prediction"].mean()
                st.metric(
                    label=f"Promedio {len(pred_df)} meses",
                    value=f"{avg_pred:.2f}%",
                )
                
                # Comparación con meta
                meta = 3.0
                if avg_pred > meta + 1:
                    st.error(f"⚠️ Por encima de meta ({meta}%)")
                elif avg_pred < meta - 1:
                    st.success(f"✅ Por debajo de meta ({meta}%)")
                else:
                    st.info(f"✅ Dentro del rango meta ({meta}±1%)")
        
        # Tabla de predicciones
        if pred_df is not None:
            st.subheader(" Tabla de Predicciones")

            # pred_df puede incluir columnas extra (p.ej. metadata). Seleccionamos
            # solo las columnas esperadas para mostrar.
            cols = [c for c in ["date", "prediction", "lower", "upper"] if c in pred_df.columns]
            display_df = pred_df[cols].copy()
            if "date" in display_df.columns:
                display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m")
            display_df = display_df.rename(
                columns={
                    "date": "Fecha",
                    "prediction": "Predicción (%)",
                    "lower": "Límite Inf.",
                    "upper": "Límite Sup.",
                }
            )
            
            st.dataframe(
                display_df.style.format({
                    "Predicción (%)": "{:.2f}",
                    "Límite Inf.": "{:.2f}",
                    "Límite Sup.": "{:.2f}",
                }),
                width="stretch",
                hide_index=True,
            )

            st.divider()
            st.subheader(" Covariables Futuras Pronosticadas")
            if covariate_forecasts is None:
                covariate_forecasts = forecast_all_covariates(df, n_steps=len(pred_df))
            cov_df = generate_future_covariates_df(
                df,
                n_months=len(pred_df),
                forecasts=covariate_forecasts,
            )

            if cov_df is not None and not cov_df.empty:
                covariate_options = [c for c in cov_df.columns if c != "date"]
                selected_covariates = st.multiselect(
                    "Selecciona covariables a visualizar",
                    options=covariate_options,
                    default=covariate_options,
                    key="covariates_selector",
                )

                fig_cov = plot_future_covariates_selected(cov_df, selected_covariates)
                if fig_cov is not None:
                    st.plotly_chart(fig_cov, width="stretch")
                else:
                    st.info("Selecciona al menos una covariable para mostrar el gráfico.")

                cov_display = cov_df.copy()
                cov_display["date"] = pd.to_datetime(cov_display["date"]).dt.strftime("%Y-%m")
                cov_display = cov_display.rename(columns={"date": "Fecha"})

                st.dataframe(
                    cov_display.style.format({
                        c: "{:.2f}" for c in cov_display.columns if c != "Fecha"
                    }),
                    width="stretch",
                    hide_index=True,
                )
    
    # ==========================================================================
    # TAB 2: DATOS HISTÓRICOS
    # ==========================================================================
    with tab2:
        st.subheader("Variables Macroeconómicas Históricas")
        
        # Selector de variable
        variables = [c for c in df.columns if c != "date"]
        selected_var = st.selectbox(
            "Selecciona una variable",
            variables,
            format_func=lambda x: VARIABLE_INFO.get(x, {"name": x})["name"],
        )
        
        # Gráfico de la variable seleccionada
        fig = plot_variable_history(df, selected_var, months_history)
        st.plotly_chart(fig, width="stretch")
        
        # Estadísticas
        col1, col2, col3, col4 = st.columns(4)
        
        recent = df[selected_var].tail(months_history)
        
        with col1:
            st.metric("Último valor", f"{df[selected_var].iloc[-1]:.2f}")
        with col2:
            st.metric("Promedio", f"{recent.mean():.2f}")
        with col3:
            st.metric("Mínimo", f"{recent.min():.2f}")
        with col4:
            st.metric("Máximo", f"{recent.max():.2f}")
        
        st.divider()
        
        # Comparación de múltiples variables
        st.subheader(" Comparación de Variables")
        
        selected_vars = st.multiselect(
            "Selecciona variables para comparar",
            variables,
            default=["Inflacion_total", "IPP"],
            format_func=lambda x: VARIABLE_INFO.get(x, {"name": x})["name"],
        )
        
        if selected_vars:
            fig = plot_multi_variable(df, selected_vars, months_history)
            st.plotly_chart(fig, width="stretch")
    
    # ==========================================================================
    # TAB 3: ANÁLISIS
    # ==========================================================================
    with tab3:
        st.subheader(" Análisis Estadístico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Matriz de correlación
            st.markdown("### Correlaciones")
            fig = plot_correlation_matrix(df)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Estadísticas descriptivas
            st.markdown("### Estadísticas Descriptivas")
            
            numeric_cols = [c for c in df.columns if c != "date"]
            stats = df[numeric_cols].describe().T
            stats.index = [VARIABLE_INFO.get(c, {"name": c})["name"][:20] for c in stats.index]
            
            st.dataframe(
                stats[["mean", "std", "min", "max"]].style.format("{:.2f}"),
                width="stretch",
            )
        
        st.divider()
        
        # Tendencias
        st.subheader(" Análisis de Tendencias")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cambio mensual promedio
            st.markdown("### Cambio Mensual Promedio (últimos 12 meses)")
            
            changes = {}
            for col in [c for c in df.columns if c != "date"]:
                recent = df[col].tail(12)
                pct_change = recent.pct_change().mean() * 100
                changes[VARIABLE_INFO.get(col, {"name": col})["name"][:15]] = pct_change
            
            changes_df = pd.DataFrame.from_dict(changes, orient="index", columns=["Cambio %"])
            changes_df = changes_df.sort_values("Cambio %", ascending=True)
            
            fig = px.bar(
                changes_df.reset_index(),
                x="Cambio %",
                y="index",
                orientation="h",
                color="Cambio %",
                color_continuous_scale="RdYlGn_r",
            )
            fig.update_layout(yaxis_title="", showlegend=False, height=300)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Volatilidad
            st.markdown("### Volatilidad (Desv. Estándar 12 meses)")
            
            volatility = {}
            for col in [c for c in df.columns if c != "date"]:
                recent = df[col].tail(12)
                vol = recent.std() / recent.mean() * 100  # Coef. de variación
                volatility[VARIABLE_INFO.get(col, {"name": col})["name"][:15]] = vol
            
            vol_df = pd.DataFrame.from_dict(volatility, orient="index", columns=["Volatilidad %"])
            vol_df = vol_df.sort_values("Volatilidad %", ascending=True)
            
            fig = px.bar(
                vol_df.reset_index(),
                x="Volatilidad %",
                y="index",
                orientation="h",
                color="Volatilidad %",
                color_continuous_scale="Oranges",
            )
            fig.update_layout(yaxis_title="", showlegend=False, height=300)
            st.plotly_chart(fig, width="stretch")
    
    # ==========================================================================
    # TAB 4: INFORMACIÓN
    # ==========================================================================
    with tab4:
        st.subheader(" Información del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Sobre el Modelo")
            st.markdown("""
            **Temporal Fusion Transformer (TFT)**
            
            El TFT es una arquitectura de deep learning diseñada específicamente para 
            pronósticos de series temporales. Combina:
            
            - **LSTM** para capturar dependencias temporales
            - **Mecanismos de atención** para identificar eventos importantes
            - **Gated Residual Networks** para selección de características
            - **Predicción cuantílica** para intervalos de confianza
            - **Decoder con covariables futuras** (sin/cos mes + IPP/TRM/Brent/FAO/Tasa/PIB)
            - **Estandarización persistente** (mean/std guardados en disco)
            
            El modelo predice 3 cuantiles (10%, 50%, 90%) para proporcionar 
            tanto el pronóstico puntual como el intervalo de confianza,
            en un horizonte directo de 6 meses.
            """)
            
            st.markdown("### Variables de Entrada")
            for var, info in VARIABLE_INFO.items():
                st.markdown(f"- **{info['name']}**: {info['description']}")
        
        with col2:
            st.markdown("### Datos del Dataset")
            
            if df is not None:
                st.info(f"**Registros:** {len(df)}")
                st.info(f"**Rango:** {df['date'].min().strftime('%Y-%m')} → {df['date'].max().strftime('%Y-%m')}")
                st.info(f"**Variables:** {len(df.columns) - 1}")
            
            st.markdown("### Configuración del Modelo")
            config_df = pd.DataFrame([
                {"Parámetro": "Ventana de entrada", "Valor": f"{CONFIG['lookback_steps']} meses"},
                {"Parámetro": "Horizonte de predicción", "Valor": f"{CONFIG['future_months']} meses"},
                {"Parámetro": "Unidades ocultas", "Valor": CONFIG['tft_units']},
                {"Parámetro": "Cabezas de atención", "Valor": CONFIG['tft_heads']},
                {"Parámetro": "Capas GRN", "Valor": CONFIG['tft_grn_layers']},
                {"Parámetro": "Dropout", "Valor": CONFIG['tft_dropout']},
            ])
            config_df["Valor"] = config_df["Valor"].astype(str)
            st.dataframe(config_df, hide_index=True, width="stretch")
            
            st.markdown("### Fuentes de Datos")
            st.markdown("""
            - **BanRep**: Banco de la República de Colombia
            - **FRED**: Federal Reserve Economic Data (Brent)
            - **FAO**: Organización de las Naciones Unidas para la Alimentación
            """)
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Sistema de Predicción de Inflación | TFT Model | 
        Última actualización: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
