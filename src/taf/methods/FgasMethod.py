"""Implementation of FGAS / FGS-Audio steganography.

Reference:
    Tian et al., "FGAS: Fixed Decoder Network-Based Audio Steganography with
    Adversarial Perturbation Generation", arXiv:2505.22266.

The transmitter (encode) produces a tiny adversarial perturbation that, when
added to the cover audio, makes a fixed, deterministically-initialized 1D-CNN
decoder output the secret bit string. The receiver (decode) reconstructs the
same decoder from the shared key and reads the bits off the sigmoid output.
Only the decoder structure and the random-seed key need to be shared.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from taf.models.SteganographyMethod import SteganographyMethod

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _he_normal(shape: Tuple[int, ...], fan_in: int, rng: np.random.Generator) -> np.ndarray:
    std = np.sqrt(2.0 / max(fan_in, 1))
    return rng.standard_normal(shape).astype(np.float32) * std


class FgasMethod(SteganographyMethod):
    """Fixed-decoder adversarial-perturbation steganography (FGAS)."""

    def __init__(
        self,
        sr: int = 16000,
        hidden_channels: int = 16,
        key: int = 1337,
        iterations: int = 200,
        epsilon: float = 1e-3,
        learning_rate: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 0.5,
        leaky_slope: float = 0.2,
    ) -> None:
        self.sr = sr
        self.hidden_channels = hidden_channels
        self.key = key
        self.iterations = iterations
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.leaky_slope = leaky_slope
        self._kernel_sizes: Tuple[int, ...] = (5, 5, 3, 3, 3)
        self._weights_cache: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------ decoder

    def _decoder_weights(self) -> List[np.ndarray]:
        if self._weights_cache is not None:
            return self._weights_cache

        rng = np.random.default_rng(self.key)
        weights: List[np.ndarray] = []
        h = self.hidden_channels
        # Channel progression: 1 -> h -> h -> h -> h -> 1
        channels = [1, h, h, h, h, 1]
        for layer_idx, k in enumerate(self._kernel_sizes):
            c_in = channels[layer_idx]
            c_out = channels[layer_idx + 1]
            fan_in = c_in * k
            w = _he_normal((k, c_in, c_out), fan_in, rng)
            b = np.zeros((c_out,), dtype=np.float32)
            weights.append(w)
            weights.append(b)
        self._weights_cache = weights
        return weights

    def _forward(self, audio_tensor, message_length: int):
        """Run the fixed decoder forward; returns sigmoid logits of shape [1, L]."""
        import tensorflow as tf

        weights = self._decoder_weights()
        x = audio_tensor  # [1, T, 1]
        eps_norm = 1e-5

        for layer_idx, k in enumerate(self._kernel_sizes):
            w = tf.constant(weights[2 * layer_idx])
            b = tf.constant(weights[2 * layer_idx + 1])
            x = tf.nn.conv1d(x, w, stride=1, padding="SAME")
            x = tf.nn.bias_add(x, b)
            # Instance normalization (per sample, per channel, over time axis)
            mean = tf.reduce_mean(x, axis=1, keepdims=True)
            var = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
            x = (x - mean) / tf.sqrt(var + eps_norm)
            x = tf.nn.leaky_relu(x, alpha=self.leaky_slope)

        # x: [1, T, 1] -> sample at L positions across the time axis. Mean-
        # pooling collapses to a near-constant when T/L is large (CLT over many
        # locally-correlated conv outputs), so the optimizer can no longer
        # differentiate the bins. Per-position read keeps each bit tied to a
        # local receptive field that is highly responsive to local
        # perturbations of delta.
        x = tf.squeeze(x, axis=-1)  # [1, T]
        t_len = tf.shape(x)[1]
        positions = tf.cast(
            tf.linspace(0.0, tf.cast(t_len - 1, tf.float32), message_length),
            tf.int32,
        )
        x = tf.gather(x, positions, axis=1)  # [1, L]
        return tf.sigmoid(x)

    # ----------------------------------------------------------------- encode

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")

        if len(message) == 0:
            return data

        cover = data.astype(np.float32).reshape(1, -1, 1)
        target = np.asarray(message, dtype=np.float32).reshape(1, -1)
        target_tf = tf.constant(target)
        cover_tf = tf.constant(cover)

        delta = tf.Variable(tf.zeros_like(cover_tf), trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        epsilon = tf.constant(self.epsilon, dtype=tf.float32)

        for _ in range(self.iterations):
            with tf.GradientTape() as tape:
                stego = tf.clip_by_value(cover_tf + delta, -1.0, 1.0)
                pred = self._forward(stego, len(message))
                pred = tf.clip_by_value(pred, 1e-6, 1.0 - 1e-6)
                loss_quality = tf.reduce_mean(tf.square(stego - cover_tf))
                loss_bits = -tf.reduce_mean(
                    target_tf * tf.math.log(pred)
                    + (1.0 - target_tf) * tf.math.log(1.0 - pred)
                )
                loss = self.alpha * loss_quality + self.beta * loss_bits

            grads = tape.gradient(loss, [delta])
            optimizer.apply_gradients(zip(grads, [delta]))
            # Project back inside the L_inf ball after every Adam step.
            delta.assign(tf.clip_by_value(delta, -epsilon, epsilon))

        stego_np = tf.clip_by_value(cover_tf + delta, -1.0, 1.0).numpy().reshape(-1)
        return stego_np.astype(data.dtype, copy=False)

    # ----------------------------------------------------------------- decode

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")

        if watermark_length == 0:
            return []

        stego = data_with_watermark.astype(np.float32).reshape(1, -1, 1)
        pred = self._forward(tf.constant(stego), watermark_length).numpy().reshape(-1)
        return [int(bit >= 0.5) for bit in pred]

    # ------------------------------------------------------------------- meta

    def type(self) -> str:
        return (
            "FGAS: Fixed Decoder Network-Based Audio Steganography with Adversarial Perturbation Generation (arXiv:2505.22266)"
        )
