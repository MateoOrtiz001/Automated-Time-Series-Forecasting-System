# Pipeline Mensual de Predicción de Inflación 📊

Sistema automatizado para la descarga de datos, predicción y mantenimiento del modelo TFT de inflación colombiana.

## 📁 Estructura de Carpetas

```
misc/
├── monthly_pipeline.py    # Script principal del pipeline
├── pipeline_state.json    # Estado persistente del pipeline
├── models/               # Modelos TFT (base y fine-tuned)
│   ├── tft_base.keras
│   └── tft_finetuned_YYYYMM_vN.keras
├── results/              # Predicciones y gráficos
│   ├── predictions_YYYYMMDD_HHMMSS.csv
│   └── predictions_plot_YYYYMMDD_HHMMSS.png
└── logs/                 # Logs de ejecución mensuales
    └── pipeline_YYYYMM.log
```

## 🚀 Uso

### Ejecución Completa (recomendado mensualmente)
```bash
python misc/monthly_pipeline.py
```

Esto realizará:
1. Descarga de datos BanRep (SUAMECA)
2. Descarga del índice FAO de precios de alimentos
3. Descarga del precio Brent (FRED)
4. Consolidación de datos
5. Fine-tuning del modelo (si han pasado 3+ meses)
6. Predicción a 12 meses
7. Generación de gráficos y CSV

### Solo Descarga de Datos
```bash
python misc/monthly_pipeline.py --download-only
```

### Solo Predicción (sin descargar)
```bash
python misc/monthly_pipeline.py --predict-only
```

### Forzar Fine-tuning
```bash
python misc/monthly_pipeline.py --finetune
```

### Omitir Descarga
```bash
python misc/monthly_pipeline.py --no-download
```

## ⚙️ Configuración

La configuración se encuentra al inicio de `monthly_pipeline.py`:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `lookback_steps` | 12 | Meses de historia para el modelo |
| `forecast_horizon` | 1 | Horizonte de predicción (paso a paso) |
| `future_months` | 12 | Meses a predecir hacia el futuro |
| `finetune_interval_months` | 3 | Intervalo para fine-tuning automático |
| `finetune_epochs` | 50 | Épocas de entrenamiento en fine-tuning |
| `finetune_lr` | 5e-4 | Learning rate para fine-tuning |

## 📈 Variables del Modelo

| Variable | Descripción |
|----------|-------------|
| `Inflacion_total` | Target - Inflación anual (%) |
| `IPP` | Índice de Precios del Productor |
| `PIB_real_trimestral_2015_AE` | PIB real trimestral |
| `Tasa_interes_colocacion_total` | Tasa de interés |
| `TRM` | Tasa de cambio COP/USD |
| `Brent` | Precio del petróleo Brent (USD) |
| `FAO` | Índice de precios de alimentos FAO |

## 📊 Resultados

### CSV de Predicciones
Columnas: `date`, `prediction`, `lower`, `upper`, `model`, `generated_at`

- `prediction`: Valor predicho (cuantil 50%)
- `lower`: Límite inferior (cuantil 10%)
- `upper`: Límite superior (cuantil 90%)

### Gráficos
- Histórico de los últimos 36 meses
- Predicción a 12 meses con intervalo de confianza (80%)

## 🔄 Estado del Pipeline

El archivo `pipeline_state.json` mantiene:
- Fecha de última descarga
- Fecha de última predicción
- Fecha de último fine-tuning
- Contador de fine-tunings
- Modelo actualmente en uso
- Historial de ejecuciones

## 🗓️ Automatización (Windows Task Scheduler)

Para ejecutar automáticamente cada mes:

1. Abrir "Task Scheduler"
2. Crear tarea básica
3. Trigger: Mensual (día 1, 8:00 AM)
4. Acción: Iniciar programa
   - Programa: `python`
   - Argumentos: `misc/monthly_pipeline.py`
   - Iniciar en: `C:\ruta\al\proyecto`

## 📝 Logs

Los logs mensuales se guardan en `misc/logs/pipeline_YYYYMM.log` con:
- Timestamp de cada operación
- Estado de descarga de cada fuente
- Métricas de fine-tuning (si aplica)
- Predicciones generadas
- Errores y advertencias

## 🔧 Troubleshooting

### Error de descarga BanRep
- Verificar conectividad a internet
- El endpoint puede estar temporalmente caído

### Error de descarga FAO/Brent
- URLs pueden haber cambiado
- Revisar `src/etl/dataExtractor.py` para actualizar

### Fine-tuning no converge
- Revisar si hay cambios bruscos en los datos
- Considerar ajustar `finetune_lr` o `finetune_epochs`

### Modelo no encontrado
- Verificar que existe `misc/models/tft_base.keras`
- Regenerar con `python test.py` si es necesario
