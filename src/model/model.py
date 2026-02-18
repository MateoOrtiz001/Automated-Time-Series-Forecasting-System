import tensorflow as tf
import keras
from tensorflow.keras.layers import (
    Layer, Dense, Add, Multiply, LayerNormalization, 
    LSTM, Input, Dropout, TimeDistributed, Reshape,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
from .customLayers import (
    GatedResidualNetwork, GatedLinearUnit, GateAddNorm, 
    MaskedMultiHeadAttention
)


class TFTModel:
    """
    Temporal Fusion Transformer simplificado.
    """
    def __init__(
        self,
        lookback_steps: int,
        forecast_horizon: int = 1,
        n_features: int = 6,
        n_future_features: int = 0,
        units: int = 32,
        num_heads: int = 2,
        num_lstm_layers: int = 1,
        num_grn_layers: int = 2,
        dropout_rate: float = 0.1,
        num_quantiles: int = 3,  # [0.1, 0.5, 0.9] para predicción probabilística
    ):
        self.lookback_steps = lookback_steps
        self.forecast_horizon = forecast_horizon
        self.n_features = n_features
        self.n_future_features = n_future_features
        self.units = units
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers
        self.num_grn_layers = num_grn_layers
        self.dropout_rate = dropout_rate
        self.num_quantiles = num_quantiles
        self.quantiles = [0.1, 0.5, 0.9][:num_quantiles]
        
        self.model = None
        self.history = None
        
    def _build_input_projection(self, inputs):
        """Proyecta las features de entrada a la dimensión del modelo."""
        # Proyección lineal de todas las features a units
        x = Dense(self.units, name="input_projection")(inputs)
        x = Dropout(self.dropout_rate)(x)
        return x
    
    def _build_grn_stack(self, x, num_layers, name_prefix="grn"):
        """Aplica múltiples capas GRN secuencialmente."""
        for i in range(num_layers):
            x = GatedResidualNetwork(self.units, self.dropout_rate)(x)
        return x
    
    def _build_temporal_processing(self, x):
        """LSTM encoder para procesamiento temporal.
        
        Returns:
            Tupla (output, (state_h, state_c)) donde output tiene shape
            (B, T, units) y los estados sirven para inicializar el decoder.
        """
        lstm_out, state_h, state_c = LSTM(
            units=self.units,
            return_sequences=True,
            return_state=True,
            dropout=self.dropout_rate,
            name="lstm_encoder"
        )(x)
        
        # GateAddNorm después del LSTM
        x = GateAddNorm(self.units)(lstm_out, x)
        return x, (state_h, state_c)
    
    def _build_attention_block(self, x):
        """Bloque de atención multi-cabeza con máscara causal."""
        # Self-attention causal
        attn_out = MaskedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.units // self.num_heads,
            dropout=self.dropout_rate
        )(query=x, value=x)
        
        # GateAddNorm después de atención
        x = GateAddNorm(self.units)(attn_out, x)
        return x
    
    def _build_output_layer(self, x):
        """Capa de salida para predicción de cuantiles (sin decoder)."""
        # GRN final antes de salida
        x = GatedResidualNetwork(self.units, self.dropout_rate)(x)
        
        # Tomar último timestep y proyectar a horizon * quantiles
        x = x[:, -1, :]  # (B, units)
        if self.forecast_horizon == 1:
            outputs = Dense(self.num_quantiles, name="quantile_output")(x)
        else:
            x = Dense(
                self.forecast_horizon * self.num_quantiles,
                name="quantile_projection",
            )(x)
            outputs = Reshape(
                (self.forecast_horizon, self.num_quantiles),
                name="quantile_output",
            )(x)
        
        return outputs
    
    def _build_decoder_output(self, encoder_out, encoder_states, future_inputs):
        """Decoder TFT para predicción multi-horizonte con covariables futuras.
        
        Arquitectura fidedigna al TFT original (simplificado):
        1. Proyección de features futuras + GRN
        2. LSTM decoder inicializado con estados del encoder
        3. Atención cruzada: decoder atiende al encoder
        4. GRN posicional + salida de cuantiles
        
        Args:
            encoder_out: Salida del encoder (B, lookback_steps, units).
            encoder_states: Tupla (state_h, state_c) del LSTM encoder.
            future_inputs: Features futuras (B, horizon, n_future_features).
        
        Returns:
            Tensor de cuantiles (B, horizon, num_quantiles).
        """
        # 1. Proyectar features futuras a la dimensión del modelo + GRN
        fut_proj = Dense(self.units, name="future_projection")(future_inputs)
        fut_proj = GatedResidualNetwork(self.units, self.dropout_rate)(fut_proj)
        
        # 2. LSTM decoder inicializado con estados del encoder
        decoder_lstm = LSTM(
            units=self.units,
            return_sequences=True,
            dropout=self.dropout_rate,
            name="lstm_decoder",
        )(fut_proj, initial_state=list(encoder_states))
        
        # Skip connection con gating
        decoder_out = GateAddNorm(self.units)(decoder_lstm, fut_proj)
        
        # 3. Atención cruzada: decoder queries → encoder keys/values
        cross_attn = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.units // self.num_heads,
            dropout=self.dropout_rate,
            name="cross_attention",
        )(query=decoder_out, key=encoder_out, value=encoder_out)
        
        # Skip connection con gating
        decoder_out = GateAddNorm(self.units)(cross_attn, decoder_out)
        
        # 4. GRN posicional
        decoder_out = GatedResidualNetwork(self.units, self.dropout_rate)(decoder_out)
        
        # 5. Salida de cuantiles por timestep
        outputs = TimeDistributed(
            Dense(self.num_quantiles), name="quantile_output"
        )(decoder_out)
        
        return outputs
    
    def build_model(self):
        """Construye el modelo TFT completo.
        
        Si n_future_features > 0 y forecast_horizon > 1, construye un modelo
        con dos entradas (past_inputs, future_inputs) y un decoder con LSTM
        inicializado desde el encoder y atención cruzada.
        """
        past_inputs = Input(shape=(self.lookback_steps, self.n_features), name="past_inputs")
        
        x = self._build_input_projection(past_inputs)
        x = self._build_grn_stack(x, self.num_grn_layers, "encoder_grn")
        x, encoder_states = self._build_temporal_processing(x)
        x = self._build_attention_block(x)
        encoder_out = self._build_grn_stack(x, 1, "post_attention_grn")
        
        if self.forecast_horizon > 1 and self.n_future_features > 0:
            future_inputs = Input(
                shape=(self.forecast_horizon, self.n_future_features),
                name="future_inputs",
            )
            outputs = self._build_decoder_output(encoder_out, encoder_states, future_inputs)
            self.model = Model(
                inputs=[past_inputs, future_inputs],
                outputs=outputs,
                name="TFT_Simplified",
            )
        else:
            outputs = self._build_output_layer(encoder_out)
            self.model = Model(
                inputs=past_inputs, outputs=outputs, name="TFT_Simplified"
            )
        
        return self.model

    @staticmethod
    def load_latest_proc_csv(proc_dir: str | Path = "data/proc") -> pd.DataFrame:
        """Carga el CSV "más reciente" dentro de `data/proc`.

        Criterios de selección (en orden de prioridad):
        1. Mayor fecha máxima en la columna 'date'
        2. Mayor número de columnas (más features)
        3. Timestamp más reciente en el nombre del archivo
        
        Nota: En entornos como Streamlit Cloud / despliegues vía git, los tiempos
        de modificación (mtime) pueden no reflejar cuál CSV contiene la data más
        actual. Por eso, usamos múltiples criterios.
        """
        proc_path = Path(proc_dir)
        if not proc_path.exists():
            raise FileNotFoundError(f"No existe el directorio: {proc_path.resolve()}")

        csvs = sorted(proc_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No hay CSVs en: {proc_path.resolve()}")

        candidates = []  # Lista de tuplas: (date_max, n_cols, timestamp_str, df, path)

        for p in csvs:
            df = pd.read_csv(p, parse_dates=["date"])
            if "date" not in df.columns or df.empty:
                continue
            df = df.sort_values("date").reset_index(drop=True)
            end = pd.to_datetime(df["date"]).max()
            n_cols = len(df.columns)
            
            # Extraer timestamp del nombre del archivo (ej: ...__20251225_212926.csv)
            name_parts = p.stem.split("__")
            timestamp_str = name_parts[-1] if name_parts else ""
            
            candidates.append((end, n_cols, timestamp_str, df, p))

        if not candidates:
            raise FileNotFoundError(f"No se pudo seleccionar un CSV válido en: {proc_path.resolve()}")

        # Ordenar por: 1) fecha máxima (desc), 2) número de columnas (desc), 3) timestamp nombre (desc)
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        
        best_df = candidates[0][3]
        return best_df

    @staticmethod
    def make_supervised_windows(
        df: pd.DataFrame,
        feature_cols: Iterable[str],
        target_col: str,
        lookback_steps: int,
        forecast_horizon: int = 1,
        future_feature_cols: Optional[Iterable[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convierte un DataFrame (ordenado por fecha) en (X, y) para TS supervised.

        - X: (N, lookback_steps, n_features)
        - y: (N,) si horizon=1, o (N, horizon) si horizon>1
        """
        feature_cols = list(feature_cols)
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' no está en df.columns")
        for c in feature_cols:
            if c not in df.columns:
                raise ValueError(f"feature '{c}' no está en df.columns")

        data_x = df[feature_cols].to_numpy(dtype=np.float32)
        data_y = df[target_col].to_numpy(dtype=np.float32)

        # Datos de features futuras (si se proporcionan)
        future_cols = list(future_feature_cols or [])
        data_future = None
        if future_cols:
            for c in future_cols:
                if c not in df.columns:
                    raise ValueError(f"future feature '{c}' no está en df.columns")
            data_future = df[future_cols].to_numpy(dtype=np.float32)

        n_total = len(df)
        max_start = n_total - lookback_steps - forecast_horizon + 1
        if max_start <= 0:
            raise ValueError(
                f"No hay suficientes filas ({n_total}) para lookback={lookback_steps} y horizon={forecast_horizon}."
            )

        X = np.stack([data_x[i : i + lookback_steps] for i in range(max_start)], axis=0)

        if forecast_horizon == 1:
            y = np.array([data_y[i + lookback_steps] for i in range(max_start)], dtype=np.float32)
        else:
            y = np.stack(
                [data_y[i + lookback_steps : i + lookback_steps + forecast_horizon] for i in range(max_start)],
                axis=0,
            ).astype(np.float32)

        # Features futuras (para decoder)
        X_future = None
        if data_future is not None and forecast_horizon > 1:
            X_future = np.stack(
                [data_future[i + lookback_steps : i + lookback_steps + forecast_horizon]
                 for i in range(max_start)],
                axis=0,
            ).astype(np.float32)

        return X, y, X_future

    @staticmethod
    def time_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Split temporal (sin shuffle): train | val | test."""
        n = len(X)
        if n < 10:
            raise ValueError("Muy pocos ejemplos para split.")

        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError("val_ratio/test_ratio demasiado grandes.")

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
        X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def time_split_finetune(X,y,val_ratio = 0.1):
        """ 
        Orden: val (primeros) | train (resto) | Sin test
        """
        n = len(X)
        if n < 10:
            raise ValueError("Muy pocos ejemplos para split.")

        n_val = int(n * val_ratio)
        if n_val < 1:
            n_val = 1
        n_train = n - n_val
        if n_train <= 0:
            raise ValueError("val_ratio demasiado grande.")

        # Val al inicio, train el resto
        X_val, y_val = X[:n_val], y[:n_val]
        X_train, y_train = X[n_val:], y[n_val:]
        
        return X_train, y_train, X_val, y_val

    @staticmethod
    def standardize_from_train(
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ):
        """Estandariza X usando media/std del train (calculadas sobre todos los timesteps)."""
        flat = X_train.reshape(-1, X_train.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < eps, 1.0, std)

        def _tx(X):
            if X is None:
                return None
            return ((X - mean) / std).astype(np.float32)

        return _tx(X_train), _tx(X_val), _tx(X_test), mean, std

    def fit_from_proc(
        self,
        proc_dir: str | Path = "data/proc",
        target_col: str = "Inflacion_total",
        feature_cols: Optional[Iterable[str]] = None,
        future_feature_cols: Optional[Iterable[str]] = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        standardize_X: bool = True,
        dropna: bool = True,
        **fit_kwargs,
    ):
        """
        Carga el CSV procesado de `data/proc`, arma ventanas y llama `fit()`.

        `fit_kwargs` se pasa a `self.fit(...)` (epochs, batch_size, patience, model_path, etc.).
        """
        df = self.load_latest_proc_csv(proc_dir)
        if dropna:
            df = df.dropna().reset_index(drop=True)

        if feature_cols is None:
            # Por defecto: todas las columnas numéricas excepto date
            feature_cols = [c for c in df.columns if c != "date"]

        X, y, X_future = self.make_supervised_windows(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            lookback_steps=self.lookback_steps,
            forecast_horizon=self.forecast_horizon,
            future_feature_cols=future_feature_cols,
        )

        X_train, y_train, X_val, y_val, X_test, y_test = self.time_split(
            X, y, val_ratio=val_ratio, test_ratio=test_ratio
        )

        Xf_train = Xf_val = Xf_test = None
        if X_future is not None and self.forecast_horizon > 1:
            Xf_train, _, Xf_val, _, Xf_test, _ = self.time_split(
                X_future, y, val_ratio=val_ratio, test_ratio=test_ratio
            )

        if standardize_X:
            X_train, X_val, X_test, mean, std = self.standardize_from_train(X_train, X_val, X_test)
            if Xf_train is not None:
                Xf_train, Xf_val, Xf_test, _, _ = self.standardize_from_train(Xf_train, Xf_val, Xf_test)

        # Asegurar compatibilidad de n_features
        if X_train.shape[-1] != self.n_features:
            self.n_features = int(X_train.shape[-1])
            # reconstruir modelo si ya existía con otro n_features
            self.model = None

        train_input = [X_train, Xf_train] if Xf_train is not None else X_train
        val_input = [X_val, Xf_val] if Xf_val is not None else X_val

        history = self.fit(
            X_train=train_input,
            y_train=y_train,
            X_val=val_input,
            y_val=y_val,
            **fit_kwargs,
        )

        return {
            "history": history,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "df": df,
            "feature_cols": list(feature_cols),
        }
    
    def quantile_loss(self, y_true, y_pred):
        """
        Pinball loss para predicción de cuantiles.
        y_true: (B,) o (B, horizon)
        y_pred: (B, num_quantiles) o (B, horizon, num_quantiles)
        """
        # Asegurar que y_true tenga shape compatible
        # - horizon=1: y_true (B,) o (B,1)
        # - horizon>1: y_true (B,H)
        if len(y_true.shape) == 2 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)

        losses = []
        for i, q in enumerate(self.quantiles):
            if len(y_pred.shape) == 2:
                pred_q = y_pred[:, i]
            else:
                pred_q = y_pred[:, :, i]
            
            error = y_true - pred_q
            loss_q = tf.maximum(q * error, (q - 1) * error)
            losses.append(loss_q)
        
        return tf.reduce_mean(tf.stack(losses, axis=-1))

    def median_mae(self, y_true, y_pred):
        """MAE calculado solo con el cuantil 0.5 (mediana) para evitar mismatch de shapes."""
        if len(y_pred.shape) == 2:
            # (B, Q)
            median_idx = 1 if self.num_quantiles >= 2 else 0
            y_hat = y_pred[:, median_idx]
        else:
            # (B, H, Q)
            median_idx = 1 if self.num_quantiles >= 2 else 0
            y_hat = y_pred[:, :, median_idx]

        if len(y_true.shape) == 2 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        return tf.reduce_mean(tf.abs(y_true - y_hat))
    
    def compile(self, learning_rate: float = 1e-3):
        """Compila el modelo con la pérdida de cuantiles."""
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=self.quantile_loss,
            metrics=[self.median_mae],
        )
        return self
    
    def fit(
        self,
        X_train: np.ndarray | list[np.ndarray],
        y_train: np.ndarray,
        X_val: np.ndarray | list[np.ndarray] = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        save_best: bool = True,
        model_path: str = "models/tft_best.keras"
    ):
        """
        Entrena el modelo.
        
        Args:
            X_train: (N, lookback_steps, n_features)
            y_train: (N,) o (N, forecast_horizon)
            X_val, y_val: datos de validación opcionales
            epochs: número máximo de épocas
            batch_size: tamaño del batch
            patience: épocas sin mejora usadas por ReduceLROnPlateau
            save_best: guardar el mejor modelo
            model_path: ruta para guardar el modelo
        """
        if self.model is None:
            self.compile()
        
        monitor = "val_loss" if X_val is not None else "loss"
        
        callbacks = [
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        if save_best:
            model_dir = os.path.dirname(model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    model_path,
                    monitor=monitor,
                    save_best_only=True,
                    verbose=1
                )
            )
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X: np.ndarray) -> dict:
        """
        Realiza predicciones.
        
        Returns:
            dict con 'median' y opcionalmente 'lower', 'upper' para intervalos
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido construido/entrenado")
        
        preds = self.model.predict(X)
        result = {"predictions": preds}

        if self.num_quantiles >= 3:
            preds_sorted = np.sort(preds, axis=-1)
            result["predictions"] = preds_sorted
            result["lower"] = preds_sorted[..., 0]      # q10
            result["median"] = preds_sorted[..., 1]     # q50
            result["upper"] = preds_sorted[..., 2]      # q90
        elif self.num_quantiles == 1:
            result["point"] = preds[..., 0]
        
        return result
    
    def save(self, path: str = "models/tft_model.keras"):
        """Guarda el modelo completo."""
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Modelo guardado en: {path}")
    
    def load(self, path: str):
        """Carga un modelo guardado."""
        self.model = keras.models.load_model(
            path,
            custom_objects={
                "quantile_loss": self.quantile_loss,
                "GatedResidualNetwork": GatedResidualNetwork,
                "GatedLinearUnit": GatedLinearUnit,
                "GateAddNorm": GateAddNorm,
                "MaskedMultiHeadAttention": MaskedMultiHeadAttention,
            }
        )
        print(f"Modelo cargado desde: {path}")
        return self
    
    def summary(self):
        """Muestra el resumen del modelo."""
        if self.model is None:
            self.build_model()
        return self.model.summary()