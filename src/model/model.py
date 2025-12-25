import tensorflow as tf
import keras
from tensorflow.keras.layers import (
    Layer, Dense, Add, Multiply, LayerNormalization, 
    LSTM, Input, Dropout, TimeDistributed
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
from customLayers import (
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
        """LSTM para procesamiento temporal."""
        # LSTM encoder
        lstm_out = LSTM(
            units=self.units,
            return_sequences=True,
            dropout=self.dropout_rate,
            name="lstm_encoder"
        )(x)
        
        # GateAddNorm después del LSTM
        x = GateAddNorm(self.units)(lstm_out, x)
        return x
    
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
        """Capa de salida para predicción de cuantiles."""
        # GRN final antes de salida
        x = GatedResidualNetwork(self.units, self.dropout_rate)(x)
        
        # Tomar solo el último timestep o aplicar TimeDistributed
        if self.forecast_horizon == 1:
            # Solo predecir el siguiente paso: tomar último timestep
            x = x[:, -1, :]  # (B, units)
            outputs = Dense(self.num_quantiles, name="quantile_output")(x)
        else:
            # Predicción multi-horizonte
            outputs = TimeDistributed(
                Dense(self.num_quantiles), 
                name="quantile_output"
            )(x)
        
        return outputs
    
    def build_model(self):
        """Construye el modelo TFT completo."""
        inputs = Input(shape=(self.lookback_steps, self.n_features), name="past_inputs")
        
        x = self._build_input_projection(inputs)
        x = self._build_grn_stack(x, self.num_grn_layers, "encoder_grn")
        x = self._build_temporal_processing(x)
        x = self._build_attention_block(x)
        x = self._build_grn_stack(x, 1, "post_attention_grn")

        outputs = self._build_output_layer(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name="TFT_Simplified")
        return self.model
    
    def quantile_loss(self, y_true, y_pred):
        """
        Pinball loss para predicción de cuantiles.
        y_true: (B,) o (B, horizon)
        y_pred: (B, num_quantiles) o (B, horizon, num_quantiles)
        """
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
    
    def compile(self, learning_rate: float = 1e-3):
        """Compila el modelo con la pérdida de cuantiles."""
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=self.quantile_loss,
            metrics=["mae"]
        )
        return self
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
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
            patience: épocas sin mejora antes de early stopping
            save_best: guardar el mejor modelo
            model_path: ruta para guardar el modelo
        """
        if self.model is None:
            self.compile()
        
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        if save_best:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    model_path,
                    monitor="val_loss" if X_val is not None else "loss",
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
            result["lower"] = preds[..., 0]      # q=0.1
            result["median"] = preds[..., 1]     # q=0.5
            result["upper"] = preds[..., 2]      # q=0.9
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