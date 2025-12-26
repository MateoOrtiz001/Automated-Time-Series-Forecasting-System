"""
Análisis de variables exógenas para predicción de inflación.

Este script evalúa qué variables tienen mayor correlación/importancia
con respecto a la inflación y recomienda cuáles mantener o descartar.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Agregar raíz del proyecto al path ANTES de otros imports
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Importar TFTModel primero para que TensorFlow se cargue antes de sklearn
from src.model.model import TFTModel

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Importar sklearn DESPUÉS de TensorFlow
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

# Configuración
PROC_DIR = ROOT_DIR / "data" / "proc"
RESULTS_DIR = ROOT_DIR / "results"
TARGET_COL = "Inflacion_total"
MAX_LAG = 12  # Máximo lag para análisis de correlación cruzada

# Descripciones de variables para interpretación
VARIABLE_DESCRIPTIONS = {
    "Inflacion_total": "Inflación total (variación anual %)",
    "IPP": "Índice de Precios del Productor (base dic/2014=100)",
    "PIB_real_trimestral_2015_AE": "PIB real trimestral con ajuste estacional (miles de millones COP)",
    "Tasa_interes_colocacion_total": "Tasa de interés de colocación total (% efectiva anual)",
    "TRM": "Tasa Representativa del Mercado (COP/USD)",
    "Brent": "Precio del petróleo Brent (USD/barril)",
    "FAO": "Índice de precios de alimentos FAO (base 2014-2016=100)",
}


def load_data() -> pd.DataFrame:
    """Carga el dataset procesado más reciente."""
    df = TFTModel.load_latest_proc_csv(PROC_DIR)
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def get_exog_cols(df: pd.DataFrame, target_col: str) -> List[str]:
    """Obtiene las columnas exógenas (todas menos date y target)."""
    return [c for c in df.columns if c not in ["date", target_col]]


# =============================================================================
# 1. CORRELACIÓN DE PEARSON (CONTEMPORÁNEA)
# =============================================================================
def analyze_pearson_correlation(df: pd.DataFrame, target_col: str, exog_cols: List[str]) -> pd.DataFrame:
    """Calcula correlación de Pearson entre exógenas y target."""
    results = []
    for col in exog_cols:
        corr, pvalue = stats.pearsonr(df[col], df[target_col])
        results.append({
            "Variable": col,
            "Correlación Pearson": corr,
            "P-value": pvalue,
            "Significativa (p<0.05)": "Sí" if pvalue < 0.05 else "No",
            "|Correlación|": abs(corr),
        })
    
    results_df = pd.DataFrame(results).sort_values("|Correlación|", ascending=False)
    return results_df


# =============================================================================
# 2. CORRELACIÓN CRUZADA (CON LAGS)
# =============================================================================
def analyze_cross_correlation(df: pd.DataFrame, target_col: str, exog_cols: List[str], max_lag: int = 12) -> pd.DataFrame:
    """Calcula correlación cruzada con diferentes lags."""
    results = []
    
    for col in exog_cols:
        best_corr = 0
        best_lag = 0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # Variable exógena adelantada (target rezagado)
                x = df[col].iloc[:lag].values
                y = df[target_col].iloc[-lag:].values
            elif lag > 0:
                # Variable exógena rezagada (target adelantado)
                x = df[col].iloc[lag:].values
                y = df[target_col].iloc[:-lag].values
            else:
                x = df[col].values
                y = df[target_col].values
            
            if len(x) > 10:
                corr, _ = stats.pearsonr(x, y)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
        
        results.append({
            "Variable": col,
            "Mejor Correlación": best_corr,
            "Mejor Lag": best_lag,
            "|Mejor Correlación|": abs(best_corr),
            "Interpretación Lag": "Variable anticipa target" if best_lag > 0 else ("Target anticipa variable" if best_lag < 0 else "Contemporánea"),
        })
    
    results_df = pd.DataFrame(results).sort_values("|Mejor Correlación|", ascending=False)
    return results_df


# =============================================================================
# 3. TEST DE CAUSALIDAD DE GRANGER (Simplificado)
# =============================================================================
def analyze_granger_causality(df: pd.DataFrame, target_col: str, exog_cols: List[str], max_lag: int = 6) -> pd.DataFrame:
    """
    Realiza test de causalidad de Granger simplificado.
    En lugar de regresiones complejas, usa correlación de cambios.
    """
    
    def manual_correlation(x, y):
        """Correlación de Pearson calculada manualmente."""
        n = len(x)
        if n < 3:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        
        if den_x == 0 or den_y == 0:
            return 0.0
        return num / (den_x * den_y)
    
    results = []
    
    target_series = list(df[target_col].values)
    target_diff = [target_series[i+1] - target_series[i] for i in range(len(target_series)-1)]
    
    for col in exog_cols:
        exog_series = list(df[col].values)
        exog_diff = [exog_series[i+1] - exog_series[i] for i in range(len(exog_series)-1)]
        
        # Buscar el mejor lag donde los cambios de exog anticipan cambios de target
        best_corr = 0.0
        best_lag = 1
        
        for lag in range(1, min(max_lag + 1, len(target_diff) - 10)):
            # exog_diff[:-lag] anticipa target_diff[lag:]
            x = exog_diff[:-lag]
            y = target_diff[lag:]
            
            if len(x) > 10 and len(x) == len(y):
                corr = manual_correlation(x, y)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
        
        # Considerar causalidad si hay correlación significativa
        granger_yes = abs(best_corr) > 0.15
        
        results.append({
            "Variable": col,
            "Granger-causa Target": "Sí" if granger_yes else "No",
            "Correlación Δ": round(best_corr, 4),
            "Lag óptimo": best_lag,
        })
    
    results_df = pd.DataFrame(results)
    # Ordenar por valor absoluto de correlación manualmente
    results_df["_abs_corr"] = results_df["Correlación Δ"].abs()
    results_df = results_df.sort_values("_abs_corr", ascending=False).drop(columns=["_abs_corr"])
    return results_df


# =============================================================================
# 4. INFORMACIÓN MUTUA
# =============================================================================
def analyze_mutual_information(df: pd.DataFrame, target_col: str, exog_cols: List[str]) -> pd.DataFrame:
    """Calcula información mutua entre exógenas y target."""
    X = df[exog_cols].values
    y = df[target_col].values
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calcular MI
    mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
    
    results = pd.DataFrame({
        "Variable": exog_cols,
        "Información Mutua": mi_scores,
    }).sort_values("Información Mutua", ascending=False)
    
    # Normalizar a porcentaje
    results["MI Normalizada (%)"] = 100 * results["Información Mutua"] / results["Información Mutua"].sum()
    
    return results


# =============================================================================
# 5. IMPORTANCIA DE FEATURES (RANDOM FOREST)
# =============================================================================
def analyze_feature_importance_rf(df: pd.DataFrame, target_col: str, exog_cols: List[str]) -> pd.DataFrame:
    """Calcula importancia de features usando Random Forest."""
    X = df[exog_cols].values
    y = df[target_col].values
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    
    results = pd.DataFrame({
        "Variable": exog_cols,
        "Importancia RF": rf.feature_importances_,
    }).sort_values("Importancia RF", ascending=False)
    
    results["Importancia RF (%)"] = 100 * results["Importancia RF"] / results["Importancia RF"].sum()
    
    return results


# =============================================================================
# 6. IMPORTANCIA DE FEATURES (GRADIENT BOOSTING)
# =============================================================================
def analyze_feature_importance_gb(df: pd.DataFrame, target_col: str, exog_cols: List[str]) -> pd.DataFrame:
    """Calcula importancia de features usando Gradient Boosting."""
    X = df[exog_cols].values
    y = df[target_col].values
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_scaled, y)
    
    results = pd.DataFrame({
        "Variable": exog_cols,
        "Importancia GB": gb.feature_importances_,
    }).sort_values("Importancia GB", ascending=False)
    
    results["Importancia GB (%)"] = 100 * results["Importancia GB"] / results["Importancia GB"].sum()
    
    return results


# =============================================================================
# 7. ANÁLISIS CONSOLIDADO Y RECOMENDACIONES
# =============================================================================
def consolidate_analysis(
    pearson_df: pd.DataFrame,
    cross_corr_df: pd.DataFrame,
    granger_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    gb_df: pd.DataFrame,
) -> pd.DataFrame:
    """Consolida todos los análisis en un ranking único."""
    
    # Merge de todos los resultados
    consolidated = pearson_df[["Variable", "|Correlación|"]].copy()
    consolidated = consolidated.merge(
        cross_corr_df[["Variable", "|Mejor Correlación|", "Mejor Lag"]], on="Variable"
    )
    consolidated = consolidated.merge(
        granger_df[["Variable", "Granger-causa Target", "Correlación Δ"]], on="Variable"
    )
    consolidated = consolidated.merge(
        mi_df[["Variable", "MI Normalizada (%)"]], on="Variable"
    )
    consolidated = consolidated.merge(
        rf_df[["Variable", "Importancia RF (%)"]], on="Variable"
    )
    consolidated = consolidated.merge(
        gb_df[["Variable", "Importancia GB (%)"]], on="Variable"
    )
    
    # Calcular score compuesto (promedio de rankings normalizados)
    # Normalizar cada métrica a [0, 1]
    consolidated["Score Correlación"] = consolidated["|Correlación|"] / consolidated["|Correlación|"].max()
    consolidated["Score Cross-Corr"] = consolidated["|Mejor Correlación|"] / consolidated["|Mejor Correlación|"].max()
    # Para Granger, usar el valor absoluto de la correlación de cambios
    consolidated["Score Granger"] = consolidated["Correlación Δ"].abs() / consolidated["Correlación Δ"].abs().max()
    consolidated["Score MI"] = consolidated["MI Normalizada (%)"] / consolidated["MI Normalizada (%)"].max()
    consolidated["Score RF"] = consolidated["Importancia RF (%)"] / consolidated["Importancia RF (%)"].max()
    consolidated["Score GB"] = consolidated["Importancia GB (%)"] / consolidated["Importancia GB (%)"].max()
    
    # Score final (promedio)
    score_cols = ["Score Correlación", "Score Cross-Corr", "Score Granger", "Score MI", "Score RF", "Score GB"]
    consolidated["SCORE FINAL"] = consolidated[score_cols].mean(axis=1)
    
    # Ordenar por score final
    consolidated = consolidated.sort_values("SCORE FINAL", ascending=False)
    
    return consolidated


def generate_recommendations(consolidated_df: pd.DataFrame, threshold: float = 0.3) -> Tuple[List[str], List[str]]:
    """Genera recomendaciones de qué variables mantener/descartar."""
    keep = consolidated_df[consolidated_df["SCORE FINAL"] >= threshold]["Variable"].tolist()
    discard = consolidated_df[consolidated_df["SCORE FINAL"] < threshold]["Variable"].tolist()
    return keep, discard


# =============================================================================
# VISUALIZACIONES
# =============================================================================
def plot_correlation_matrix(df: pd.DataFrame, target_col: str, exog_cols: List[str], save_path: Path):
    """Genera matriz de correlación."""
    cols = [target_col] + exog_cols
    corr_matrix = df[cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0, fmt=".2f",
                square=True, linewidths=0.5)
    plt.title("Matriz de Correlación: Inflación vs Variables Exógenas")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(consolidated_df: pd.DataFrame, save_path: Path):
    """Gráfico de barras con scores de importancia."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Score Final
    ax1 = axes[0, 0]
    data = consolidated_df.sort_values("SCORE FINAL", ascending=True)
    ax1.barh(data["Variable"], data["SCORE FINAL"], color="steelblue")
    ax1.set_xlabel("Score Final")
    ax1.set_title("Score Final (Combinado)")
    ax1.axvline(x=0.3, color="red", linestyle="--", label="Umbral recomendado (0.3)")
    ax1.legend()
    
    # 2. Correlación Pearson
    ax2 = axes[0, 1]
    data = consolidated_df.sort_values("|Correlación|", ascending=True)
    ax2.barh(data["Variable"], data["|Correlación|"], color="coral")
    ax2.set_xlabel("|Correlación Pearson|")
    ax2.set_title("Correlación con Inflación")
    
    # 3. Importancia RF
    ax3 = axes[1, 0]
    data = consolidated_df.sort_values("Importancia RF (%)", ascending=True)
    ax3.barh(data["Variable"], data["Importancia RF (%)"], color="forestgreen")
    ax3.set_xlabel("Importancia (%)")
    ax3.set_title("Importancia Random Forest")
    
    # 4. Información Mutua
    ax4 = axes[1, 1]
    data = consolidated_df.sort_values("MI Normalizada (%)", ascending=True)
    ax4.barh(data["Variable"], data["MI Normalizada (%)"], color="darkorange")
    ax4.set_xlabel("Información Mutua (%)")
    ax4.set_title("Información Mutua")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_series_comparison(df: pd.DataFrame, target_col: str, exog_cols: List[str], save_path: Path):
    """Gráfico de series temporales normalizadas."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Normalizar para visualización
    scaler = StandardScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(df[[target_col] + exog_cols]),
        columns=[target_col] + exog_cols
    )
    normalized["date"] = df["date"].values
    
    # Graficar target más grueso
    ax.plot(normalized["date"], normalized[target_col], label=target_col, linewidth=3, color="black")
    
    # Graficar exógenas
    colors = plt.cm.tab10(np.linspace(0, 1, len(exog_cols)))
    for col, color in zip(exog_cols, colors):
        ax.plot(normalized["date"], normalized[col], label=col, alpha=0.7, color=color)
    
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor Normalizado (Z-score)")
    ax.set_title("Series Temporales Normalizadas: Inflación vs Exógenas")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("ANÁLISIS DE VARIABLES EXÓGENAS PARA PREDICCIÓN DE INFLACIÓN")
    print("=" * 70)
    
    # Cargar datos
    df = load_data()
    exog_cols = get_exog_cols(df, TARGET_COL)
    
    print(f"\nDataset: {len(df)} filas")
    print(f"Rango: {df['date'].min()} → {df['date'].max()}")
    print(f"Target: {TARGET_COL}")
    print(f"Variables exógenas ({len(exog_cols)}):")
    for col in exog_cols:
        desc = VARIABLE_DESCRIPTIONS.get(col, col)
        print(f"   • {col}: {desc}")
    
    # Estadísticas básicas del target
    print(f"\n{'='*70}")
    print("ESTADÍSTICAS DEL TARGET (Inflación)")
    print("="*70)
    print(df[TARGET_COL].describe())
    
    # 1. Correlación de Pearson
    print(f"\n{'='*70}")
    print("1. CORRELACIÓN DE PEARSON (CONTEMPORÁNEA)")
    print("="*70)
    pearson_df = analyze_pearson_correlation(df, TARGET_COL, exog_cols)
    print(pearson_df.to_string(index=False))
    
    # 2. Correlación cruzada
    print(f"\n{'='*70}")
    print("2. CORRELACIÓN CRUZADA (CON LAGS)")
    print("="*70)
    cross_corr_df = analyze_cross_correlation(df, TARGET_COL, exog_cols, MAX_LAG)
    print(cross_corr_df.to_string(index=False))
    
    # 3. Causalidad de Granger (Simplificado - sin statsmodels)
    print(f"\n{'='*70}")
    print("3. TEST DE CAUSALIDAD DE GRANGER (Simplificado)")
    print("="*70)
    sys.stdout.flush()
    granger_df = analyze_granger_causality(df, TARGET_COL, exog_cols)
    sys.stdout.flush()
    print(granger_df.to_string(index=False))
    sys.stdout.flush()
    
    # 4. Información Mutua
    print(f"\n{'='*70}")
    print("4. INFORMACIÓN MUTUA")
    print("="*70)
    mi_df = analyze_mutual_information(df, TARGET_COL, exog_cols)
    print(mi_df.to_string(index=False))
    
    # 5. Importancia Random Forest
    print(f"\n{'='*70}")
    print("5. IMPORTANCIA DE FEATURES (RANDOM FOREST)")
    print("="*70)
    rf_df = analyze_feature_importance_rf(df, TARGET_COL, exog_cols)
    print(rf_df.to_string(index=False))
    
    # 6. Importancia Gradient Boosting
    print(f"\n{'='*70}")
    print("6. IMPORTANCIA DE FEATURES (GRADIENT BOOSTING)")
    print("="*70)
    gb_df = analyze_feature_importance_gb(df, TARGET_COL, exog_cols)
    print(gb_df.to_string(index=False))
    
    # 7. Análisis consolidado
    print(f"\n{'='*70}")
    print("7. ANÁLISIS CONSOLIDADO Y RANKING FINAL")
    print("="*70)
    consolidated_df = consolidate_analysis(pearson_df, cross_corr_df, granger_df, mi_df, rf_df, gb_df)
    
    # Mostrar tabla resumida
    display_cols = ["Variable", "|Correlación|", "|Mejor Correlación|", "Mejor Lag",
                    "Granger-causa Target", "MI Normalizada (%)", 
                    "Importancia RF (%)", "Importancia GB (%)", "SCORE FINAL"]
    print(consolidated_df[display_cols].to_string(index=False))
    
    # 8. Recomendaciones
    print(f"\n{'='*70}")
    print("8. RECOMENDACIONES")
    print("="*70)
    
    keep, discard = generate_recommendations(consolidated_df, threshold=0.3)
    
    print("\n✅ VARIABLES RECOMENDADAS PARA MANTENER (Score >= 0.3):")
    for i, var in enumerate(keep, 1):
        score = consolidated_df[consolidated_df["Variable"] == var]["SCORE FINAL"].values[0]
        print(f"   {i}. {var} (Score: {score:.3f})")
    
    print("\n⚠️ VARIABLES CANDIDATAS A DESCARTAR (Score < 0.3):")
    if discard:
        for i, var in enumerate(discard, 1):
            score = consolidated_df[consolidated_df["Variable"] == var]["SCORE FINAL"].values[0]
            print(f"   {i}. {var} (Score: {score:.3f})")
    else:
        print("   Ninguna - todas las variables tienen score >= 0.3")
    
    # 9. Interpretación detallada
    print(f"\n{'='*70}")
    print("9. INTERPRETACIÓN DETALLADA")
    print("="*70)
    
    for _, row in consolidated_df.iterrows():
        var = row["Variable"]
        desc = VARIABLE_DESCRIPTIONS.get(var, "")
        print(f"\n📊 {var}")
        if desc:
            print(f"   ({desc})")
        print(f"   • Correlación Pearson: {row['|Correlación|']:.3f}")
        print(f"   • Mejor correlación cruzada: {row['|Mejor Correlación|']:.3f} (lag={int(row['Mejor Lag'])} meses)")
        lag_interp = "anticipa" if row['Mejor Lag'] > 0 else ("sigue a" if row['Mejor Lag'] < 0 else "contemporánea con")
        if row['Mejor Lag'] != 0:
            print(f"     → Esta variable {lag_interp} la inflación por {abs(int(row['Mejor Lag']))} mes(es)")
        print(f"   • Granger-causa inflación: {row['Granger-causa Target']}")
        print(f"   • Información Mutua: {row['MI Normalizada (%)']:.1f}%")
        print(f"   • Importancia RF: {row['Importancia RF (%)']:.1f}%")
        print(f"   • Importancia GB: {row['Importancia GB (%)']:.1f}%")
        print(f"   → SCORE FINAL: {row['SCORE FINAL']:.3f}")
    
    # 10. Guardar resultados
    print(f"\n{'='*70}")
    print("10. GUARDANDO RESULTADOS")
    print("="*70)
    
    # CSV con análisis consolidado
    output_csv = RESULTS_DIR / "feature_analysis_results.csv"
    consolidated_df.to_csv(output_csv, index=False)
    print(f"✓ Análisis guardado en: {output_csv}")
    
    # Gráficos
    plot_correlation_matrix(df, TARGET_COL, exog_cols, RESULTS_DIR / "feature_correlation_matrix.png")
    print(f"✓ Matriz de correlación guardada")
    
    plot_feature_importance(consolidated_df, RESULTS_DIR / "feature_importance.png")
    print(f"✓ Gráfico de importancia guardado")
    
    plot_time_series_comparison(df, TARGET_COL, exog_cols, RESULTS_DIR / "feature_time_series.png")
    print(f"✓ Comparación de series temporales guardada")
    
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)
    
    return consolidated_df


if __name__ == "__main__":
    results = main()
