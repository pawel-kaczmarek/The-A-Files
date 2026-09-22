"""Learnable-embedding audio watermarking with GA optimization.

Reference:
    J. Nayeem, H.-B. Lee, and Y.-H. Seo, "Robust audio watermarking
    with learnable embedding technique and genetic optimization",
    Digital Signal Processing, vol. 183, article 106372, 2026.
    https://doi.org/10.1016/j.dsp.2026.106372

The paper describes a trained neural watermarking pipeline, but does not
publish pretrained weights or exact channel widths for every learned module.
This implementation keeps to the paper's public signal flow and equations:

* Eq. (2): binary watermark -> continuous representation.
* Eq. (3): representation -> waveform-length upscaled watermark.
* Eq. (5): perceptual mask modulation.
* Eq. (6): GA watermark fitness by average energy.
* Section 3.3: scaled residual embedding into the carrier audio.
* Eqs. (9)-(10): pooled extractor output thresholded to hard bits.

Smallest implementation assumption:
    In the absence of trained network weights, the learned modules are
    instantiated as deterministic minimal operators: an upscaled zero-mean
    waveform representation, an all-ones mask, residual addition, and
    correlation-style adaptive pooling for extraction.
"""
from __future__ import annotations

from typing import List

import numpy as np

from taf.models.SteganographyMethod import SteganographyMethod


def _validate_audio(data: np.ndarray) -> None:
    if not isinstance(data, np.ndarray):
        raise TypeError("audio data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("audio data must be a one-dimensional mono signal")
    if data.size == 0:
        raise ValueError("audio data must not be empty")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("audio data must contain numeric samples")
    if not np.all(np.isfinite(data.astype(np.float64))):
        raise ValueError("audio data must contain only finite samples")


def _validate_message_bits(message: List[int]) -> List[int]:
    bits: List[int] = []
    for bit in message:
        try:
            value = int(bit)
        except (TypeError, ValueError) as error:
            raise ValueError("message bits must contain only 0 and 1") from error
        if value not in (0, 1) or bit != value:
            raise ValueError("message bits must contain only 0 and 1")
        bits.append(value)
    return bits


def _to_float_audio(data: np.ndarray) -> tuple[np.ndarray, int | None]:
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        peak = max(abs(info.min), info.max)
        return data.astype(np.float64) / peak, peak
    return data.astype(np.float64, copy=True), None


def _restore_dtype(data: np.ndarray, dtype: np.dtype, integer_peak: int | None) -> np.ndarray:
    if integer_peak is None:
        return data.astype(dtype, copy=False)

    info = np.iinfo(dtype)
    restored = np.rint(np.clip(data, -1.0, 1.0) * integer_peak)
    return np.clip(restored, info.min, info.max).astype(dtype)


class LearnableEmbeddingGaMethod(SteganographyMethod):
    """Robust audio watermarking with learnable embedding and GA optimization."""

    def __init__(
            self,
            scaling_factor: float = 0.9,
            embedding_strength: float = 0.005,
            population_size: int = 10,
            mutation_rate: float = 0.1,
            mutation_std: float = 0.05,
            seed: int = 42,
            min_samples_per_bit: int = 16,
    ) -> None:
        if scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive")
        if embedding_strength <= 0:
            raise ValueError("embedding_strength must be positive")
        if population_size < 1:
            raise ValueError("population_size must be positive")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if mutation_std < 0:
            raise ValueError("mutation_std must be non-negative")
        if min_samples_per_bit < 2:
            raise ValueError("min_samples_per_bit must be at least 2")

        # Paper defaults: fixed scaling factor 0.9, population 10, mutation
        # rate 0.1, seed 42, and 36-bit payloads in experiments. The raw
        # waveform perturbation gain is not specified; 0.01 keeps normalized
        # residual additions small while making direct extraction observable.
        self.scaling_factor = scaling_factor
        self.embedding_strength = embedding_strength
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.seed = seed
        self.min_samples_per_bit = min_samples_per_bit

    # -------------------------------------------------------------- validation

    def _validate_capacity(self, sample_count: int, watermark_length: int, name: str) -> None:
        capacity = sample_count // self.min_samples_per_bit
        if watermark_length > capacity:
            raise ValueError(
                f"{name} too long for cover audio: {watermark_length} > {capacity} bits"
            )

    @staticmethod
    def _segment_bounds(sample_count: int, bit_count: int) -> np.ndarray:
        return np.linspace(0, sample_count, bit_count + 1, dtype=np.int64)

    # ------------------------------------------------------------- paper steps

    def _bit_carrier(self, length: int, bit: int, segment_index: int) -> np.ndarray:
        """Eq. (2) minimal vector transform for one bit.

        The paper converts bits into a continuous latent representation. With
        no learned weights supplied, a key-derived bipolar pseudo-noise
        sequence stands in for it: it is zero-mean, so residual embedding adds
        no DC bias, and noise-like, so it does not concentrate the watermark
        at one frequency. An alternating +1/-1 sequence, used previously, is a
        tone at half the sampling rate and is both audible and trivial to
        strip with a low-pass filter.
        """
        rng = np.random.default_rng((self.seed, segment_index))
        carrier = rng.integers(0, 2, length).astype(np.float64) * 2.0 - 1.0
        if bit == 0:
            carrier *= -1.0
        carrier -= float(np.mean(carrier))
        rms = float(np.sqrt(np.mean(np.square(carrier))))
        if rms == 0.0:
            raise ValueError("audio segment too short for watermark carrier")
        return carrier / rms

    def _upscale_watermark(self, bits: List[int], sample_count: int) -> np.ndarray:
        """Eq. (3): expand the bit representation to audio length T."""
        upscaled = np.zeros(sample_count, dtype=np.float64)
        bounds = self._segment_bounds(sample_count, len(bits))

        for idx, bit in enumerate(bits):
            start, stop = int(bounds[idx]), int(bounds[idx + 1])
            upscaled[start:stop] = self._bit_carrier(stop - start, bit, idx)

        return upscaled

    @staticmethod
    def _mask(audio: np.ndarray) -> np.ndarray:
        """Eq. (5) mask predictor.

        The paper's perceptual mask is a learned CNN. Since the paper does not
        provide its trained weights, the conservative deployable mask is the
        identity mask M=1, preserving the described modulation point without
        adding a hand-crafted psychoacoustic model.
        """
        return np.ones_like(audio, dtype=np.float64)

    def _evolve_watermark(self, watermark: np.ndarray) -> np.ndarray:
        """Eq. (6): GA evolution scoring candidates by recoverable energy.

        Scoring candidates by raw average energy, as a literal reading of the
        equation suggests, rewards the mutation noise itself: the selected
        candidate drifts away from the carrier the extractor correlates
        against, costing bits at low embedding strength. Projecting onto the
        intended watermark keeps the fitness an energy measure while counting
        only the component the extractor can actually see.
        """
        if self.population_size == 1 or self.mutation_std == 0.0:
            return watermark

        rng = np.random.default_rng(self.seed)
        population = [watermark]

        for _ in range(self.population_size - 1):
            candidate = watermark.copy()
            mutation_mask = rng.random(candidate.shape) < self.mutation_rate
            mutation = rng.normal(0.0, self.mutation_std, size=candidate.shape)
            candidate[mutation_mask] += mutation[mutation_mask]
            # The paper does not state mutation bounds. Clipping keeps the
            # evolved continuous watermark in the same normalized range as the
            # upscaled representation.
            population.append(np.clip(candidate, -1.0, 1.0))

        fitness = [float(np.mean(candidate * watermark)) for candidate in population]
        return population[int(np.argmax(fitness))]

    # ----------------------------------------------------------------- encode

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        _validate_audio(data)
        bits = _validate_message_bits(message)
        if len(bits) == 0:
            return data.copy()

        self._validate_capacity(len(data), len(bits), "message")

        audio, integer_peak = _to_float_audio(data)

        # Eqs. (2)-(3): binary watermark -> continuous upscaled watermark.
        watermark = self._upscale_watermark(bits, len(audio))

        # Eq. (6): GA-selected watermark candidate.
        evolved_watermark = self._evolve_watermark(watermark)

        # Eq. (5) and Section 3.3: mask modulation and scaled residual
        # embedding. The carrier decoder residual skip is represented by adding
        # the watermark residual back to the original waveform.
        masked_watermark = self._mask(audio) * evolved_watermark
        residual = self.scaling_factor * self.embedding_strength * masked_watermark
        stego = np.clip(audio + residual, -1.0, 1.0)

        return _restore_dtype(stego, data.dtype, integer_peak)

    # ----------------------------------------------------------------- decode

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        _validate_audio(data_with_watermark)
        if not isinstance(watermark_length, (int, np.integer)):
            raise TypeError("watermark_length must be an integer")
        if watermark_length < 0:
            raise ValueError("watermark_length must be non-negative")
        if watermark_length == 0:
            return []

        self._validate_capacity(len(data_with_watermark), watermark_length, "watermark")

        audio, _ = _to_float_audio(data_with_watermark)
        bounds = self._segment_bounds(len(audio), watermark_length)
        decoded: List[int] = []

        for idx in range(watermark_length):
            start, stop = int(bounds[idx]), int(bounds[idx + 1])
            segment = audio[start:stop]
            # Eqs. (9)-(10): minimal convolution/pooling extractor. Correlation
            # with the bit-1 representation is the deterministic equivalent of
            # the pooled decoder logit, followed by sigmoid thresholding.
            carrier = self._bit_carrier(stop - start, 1, idx)
            # Pre-whiten both sides with a first difference: audio energy is
            # concentrated at low frequencies while the carrier is white, so
            # correlating the raw segment reads mostly host interference.
            pooled_score = float(np.mean(np.diff(segment) * np.diff(carrier)))
            decoded.append(1 if pooled_score >= 0.0 else 0)

        return decoded

    # ------------------------------------------------------------------- meta

    def type(self) -> str:
        return "Learnable embedding with GA audio watermarking method"
