from tensorflow.keras.layers import Layer, Dense, Add, Multiply, LayerNormalization, LSTM, Input
from tensorflow.keras.layers import Dropout, TimeDistributed, MultiHeadAttention, Embedding
from tensorflow.keras import ops
import keras
import tensorflow as tf
import numpy as np


class VariableSelection(Layer):
    """
    inputs: dict {name: Tensor}, cada Tensor de shape (B, T) para cat o (B, T) num.
    """
    def __init__(self, cat_vocab_sizes, num_feature_names, units, dropout_rate):
        super().__init__()
        self.units = units
        self.embeddings = {
            name: Embedding(input_dim=vocab, output_dim=units, name=f"emb_{name}")
            for name, vocab in cat_vocab_sizes.items()
        }
        self.proj = {
            name: Dense(units, name=f"proj_{name}")
            for name in num_feature_names
        }
        self.feature_names = list(cat_vocab_sizes.keys()) + list(num_feature_names)
        self.grns = [GatedResidualNetwork(units, dropout_rate) for _ in self.feature_names]
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = Dense(units=len(self.feature_names), activation="softmax")
    def call(self, inputs):
        encoded = []
        for name in self.feature_names:
            if name in self.embeddings:
                # clip por seguridad
                vocab_limit = self.embeddings[name].input_dim - 1
                z = self.embeddings[name](ops.clip(inputs[name], 0, vocab_limit))
            else:
                z = ops.expand_dims(inputs[name], -1)
                z = self.proj[name](z)
            encoded.append(z)  # cada z: (B, T, units)
        # atención sobre features
        concat_all = ops.concatenate(encoded, axis=-1)            # (B, T, F*units)
        scores = self.softmax(self.grn_concat(concat_all))        # (B, T, F)
        scores = ops.expand_dims(scores, axis=-1)                 # (B, T, F, 1)
        stacked = ops.stack(encoded, axis=2)                      # (B, T, F, units)
        weighted = scores * stacked
        return ops.sum(weighted, axis=2)                          # (B, T, units)

class MaskedMultiHeadAttention(Layer):
    def __init__(self, num_heads, key_dim, dropout=0.0):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
    def call(self, query, value, training=None):
        seq_len = tf.shape(query)[1]
        # máscara causal (T, T)
        causal = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        # Keras espera (B, heads, Tq, Tv) o (B, Tq, Tv); usamos broadcast
        attn_mask = causal[None, :, :]  # (1, T, T)
        return self.mha(query=query, value=value, attention_mask=attn_mask, training=training)

class StaticCovariateEncoder(Layer):
    def __init__(self, cat_vocab_sizes, num_feature_names, units, dropout_rate):
        super().__init__()
        self.units = units
        self.embeddings = {
            name: Embedding(input_dim=vocab, output_dim=units, name=f"emb_static_{name}")
            for name, vocab in cat_vocab_sizes.items()
        }
        self.proj = {
            name: Dense(units, name=f"proj_static_{name}")
            for name in num_feature_names
        }
        self.grn_cs = GatedResidualNetwork(units, dropout_rate)  # para variable selection conditioning
        self.grn_ce = GatedResidualNetwork(units, dropout_rate)  # para inicializar LSTM (h/c)
        self.grn_cd = GatedResidualNetwork(units, dropout_rate)  # para fusion/attention
        self.feature_names = list(cat_vocab_sizes.keys()) + list(num_feature_names)
    def call(self, inputs):
        encoded = []
        for name in self.feature_names:
            if name in self.embeddings:
                vocab_limit = self.embeddings[name].input_dim - 1
                z = self.embeddings[name](ops.clip(inputs[name], 0, vocab_limit))
            else:
                z = ops.expand_dims(inputs[name], -1)
                z = self.proj[name](z)
            encoded.append(z)  # cada z: (B, units) si input (B,)
        s = ops.concatenate(encoded, axis=-1)
        c_s = self.grn_cs(s)
        c_e = self.grn_ce(s)
        c_d = self.grn_cd(s)
        return c_s, c_e, c_d

class GatedLinearUnit(Layer):
    def __init__(self, units):
        super().__init__()
        self.linear = Dense(units)
        self.sigmoid = Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

    # Remove build warnings
    def build(self):
        self.built = True


class GatedResidualNetwork(Layer):
    def __init__(self, units, dropout_rate):
        super().__init__()
        self.units = units
        self.elu_dense = Dense(units, activation="elu")
        self.linear_dense = Dense(units)
        self.dropout = Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = LayerNormalization()
        self.project = Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x

    # Remove build warnings
    def build(self):
        self.built = True


class GateAddNorm(Layer):
    """Aplica GLU + residual + LayerNorm."""
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.glu = GatedLinearUnit(units)
        self.layer_norm = LayerNormalization()
        
    def call(self, x, residual):
        gated = self.glu(x)
        return self.layer_norm(residual + gated)

