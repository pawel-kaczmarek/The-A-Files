from __future__ import annotations

import numpy as np
import pytest

from taf.methods.LearnableEmbeddingGaMethod import LearnableEmbeddingGaMethod
from taf.methods.factory import SteganographyMethodFactory
from taf.models.types import MethodType


def _synthetic_audio() -> np.ndarray:
    rng = np.random.default_rng(seed=20260705)
    t = np.arange(16000) / 16000.0
    sine = 0.4 * np.sin(2 * np.pi * 440.0 * t)
    noise = rng.normal(0.0, 0.002, size=t.shape)
    return (sine + noise).astype(np.float32)


def test_learnable_embedding_ga_method_can_be_created() -> None:
    method = SteganographyMethodFactory.get(16000, MethodType.LEARNABLE_EMBEDDING_GA_METHOD)

    assert isinstance(method, LearnableEmbeddingGaMethod)
    assert "Learnable embedding" in method.type()


def test_learnable_embedding_ga_encode_returns_same_shape() -> None:
    cover = _synthetic_audio()
    message = [1, 0, 1, 1, 0, 0, 1, 0]
    method = LearnableEmbeddingGaMethod()

    stego = method.encode(cover.copy(), message)

    assert isinstance(stego, np.ndarray)
    assert stego.shape == cover.shape
    assert stego.dtype == cover.dtype


def test_learnable_embedding_ga_decode_returns_requested_number_of_bits() -> None:
    cover = _synthetic_audio()
    message = [1, 0, 1, 1, 0, 0, 1, 0]
    method = LearnableEmbeddingGaMethod()

    stego = method.encode(cover.copy(), message)
    decoded = method.decode(stego, len(message))

    assert len(decoded) == len(message)
    assert all(bit in (0, 1) for bit in decoded)


def test_learnable_embedding_ga_roundtrip_on_synthetic_audio() -> None:
    rng = np.random.default_rng(seed=20260706)
    cover = _synthetic_audio()
    message = [int(bit) for bit in rng.integers(0, 2, size=36)]
    method = LearnableEmbeddingGaMethod()

    stego = method.encode(cover.copy(), message)
    decoded = method.decode(stego, len(message))

    assert decoded == message


def test_learnable_embedding_ga_rejects_non_numpy_audio() -> None:
    method = LearnableEmbeddingGaMethod()

    with pytest.raises(TypeError, match="audio data"):
        method.encode([0.0, 0.1], [1])  # type: ignore[arg-type]


def test_learnable_embedding_ga_rejects_non_binary_message() -> None:
    method = LearnableEmbeddingGaMethod()

    with pytest.raises(ValueError, match="message bits"):
        method.encode(np.zeros(160, dtype=np.float32), [0, 2, 1])


def test_learnable_embedding_ga_rejects_too_large_payload() -> None:
    method = LearnableEmbeddingGaMethod(min_samples_per_bit=16)

    with pytest.raises(ValueError, match="message too long"):
        method.encode(np.zeros(32, dtype=np.float32), [1, 0, 1])


def test_learnable_embedding_ga_rejects_negative_watermark_length() -> None:
    method = LearnableEmbeddingGaMethod()

    with pytest.raises(ValueError, match="watermark_length"):
        method.decode(np.zeros(160, dtype=np.float32), -1)