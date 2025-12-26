"""
Dashboard de Predicción de Inflación - Colombia
================================================

Dashboard interactivo en Streamlit para visualizar:
- Predicciones de inflación a 12 meses (modelo TFT)
- Datos históricos de variables macroeconómicas
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

from src.model.model import TFTModel

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

# Configuración del modelo
CONFIG = {
    "lookback_steps": 12,
    "forecast_horizon": 1,
    "target_col": "Inflacion_total",
    "future_months": 12,
    "tft_units": 74,
    "tft_heads": 2,
    "tft_lstm_layers": 1,
    "tft_grn_layers": 2,
    "tft_dropout": 0.1,
}

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
@st.cache_data(ttl=3600)
def load_data():
    """Carga los datos procesados más recientes."""
    try:
        df = TFTModel.load_latest_proc_csv(DATA_PROC_DIR)
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None


@st.cache_resource
def load_model():
    """Carga el modelo TFT más reciente."""
    # Buscar modelo fine-tuned primero
    model_path = None
    
    if MISC_MODELS_DIR.exists():
        finetuned = sorted(MISC_MODELS_DIR.glob("tft_finetuned_*.keras"), reverse=True)
        if finetuned:
            model_path = finetuned[0]
    
    if model_path is None:
        model_path = MISC_MODELS_DIR / "tft_base.keras"
        if not model_path.exists():
            model_path = MODELS_DIR / "tft_best.keras"
    
    if not model_path.exists():
        return None, None
    
    # Cargar datos para obtener n_features
    df = load_data()
    if df is None:
        return None, None
    
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
    
    return tft, model_path.name


def load_pipeline_state():
    """Carga el estado del pipeline."""
    state_file = MISC_DIR / "pipeline_state.json"
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_latest_predictions():
    """Carga las predicciones más recientes."""
    if not MISC_RESULTS_DIR.exists():
        return None
    
    pred_files = sorted(MISC_RESULTS_DIR.glob("predictions_*.csv"), reverse=True)
    if not pred_files:
        return None
    
    df = pd.read_csv(pred_files[0], parse_dates=["date"])
    return df


def generate_predictions(model, df, n_months=12):
    """Genera predicciones para los próximos n meses."""
    feature_cols = [c for c in df.columns if c != "date"]
    target_idx = feature_cols.index(CONFIG["target_col"])
    
    # Calcular estadísticas de normalización
    X, y = TFTModel.make_supervised_windows(
        df=df,
        feature_cols=feature_cols,
        target_col=CONFIG["target_col"],
        lookback_steps=CONFIG["lookback_steps"],
        forecast_horizon=CONFIG["forecast_horizon"],
    )
    X_train, _, _, _, _, _ = TFTModel.time_split(X, y)
    _, _, _, mean, std = TFTModel.standardize_from_train(X_train)
    
    # Preparar datos
    data_features = df[feature_cols].to_numpy(dtype=np.float32)
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
    
    return pd.DataFrame(predictions)


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
    
    # Intervalo de confianza
    if "lower" in df_pred.columns and df_pred["lower"].notna().any():
        fig.add_trace(go.Scatter(
            x=pd.concat([df_pred["date"], df_pred["date"][::-1]]),
            y=pd.concat([df_pred["upper"], df_pred["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(231, 76, 60, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="IC 80%",
            showlegend=True,
        ))
    
    # Línea vertical de corte (convertir a string para compatibilidad con plotly)
    last_date = df_hist["date"].iloc[-1]
    if hasattr(last_date, 'strftime'):
        last_date = last_date.strftime('%Y-%m-%d')
    
    fig.add_vline(
        x=last_date,
        line_dash="dot",
        line_color="gray",
        annotation_text="Último dato real",
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


# =============================================================================
# APLICACIÓN PRINCIPAL
# =============================================================================
def main():
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
    
    # Cargar datos
    df = load_data()
    if df is None:
        st.error("No se pudieron cargar los datos. Verifica que existan archivos en data/proc/")
        return
    
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
            st.subheader("Predicción de Inflación a 12 Meses")
            
            # Cargar o generar predicciones
            model, model_name = load_model()
            
            if model is None:
                st.warning("No se encontró modelo entrenado. Entrena el modelo primero.")
                pred_df = load_latest_predictions()
            elif st.session_state.get("regenerate", False):
                with st.spinner("Generando predicciones..."):
                    pred_df = generate_predictions(model, df)
                    st.session_state["regenerate"] = False
                    st.success(" Predicciones generadas")
            else:
                # Intentar cargar predicciones existentes o generar nuevas
                pred_df = load_latest_predictions()
                if pred_df is None and model is not None:
                    with st.spinner("Generando predicciones..."):
                        pred_df = generate_predictions(model, df)
            
            if pred_df is not None:
                fig = plot_inflation_forecast(df, pred_df, months_history)
                if not show_confidence:
                    fig.data = [t for t in fig.data if t.name != "IC 80%"]
                st.plotly_chart(fig, use_container_width=True)
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
                    label="Promedio 12 meses",
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
            
            display_df = pred_df.copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m")
            display_df.columns = ["Fecha", "Predicción (%)", "Límite Inf.", "Límite Sup."]
            
            st.dataframe(
                display_df.style.format({
                    "Predicción (%)": "{:.2f}",
                    "Límite Inf.": "{:.2f}",
                    "Límite Sup.": "{:.2f}",
                }),
                use_container_width=True,
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
        st.plotly_chart(fig, use_container_width=True)
        
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
            st.plotly_chart(fig, use_container_width=True)
    
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
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Estadísticas descriptivas
            st.markdown("### Estadísticas Descriptivas")
            
            numeric_cols = [c for c in df.columns if c != "date"]
            stats = df[numeric_cols].describe().T
            stats.index = [VARIABLE_INFO.get(c, {"name": c})["name"][:20] for c in stats.index]
            
            st.dataframe(
                stats[["mean", "std", "min", "max"]].style.format("{:.2f}"),
                use_container_width=True,
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
            st.plotly_chart(fig, use_container_width=True)
        
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
            st.plotly_chart(fig, use_container_width=True)
    
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
            
            El modelo predice 3 cuantiles (10%, 50%, 90%) para proporcionar 
            tanto el pronóstico puntual como el intervalo de confianza.
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
            st.dataframe(config_df, hide_index=True, use_container_width=True)
            
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
