"""AudioSeal neural watermarking.

Reference:
    San Roman, R., Fernandez, P., Elsahar, H., Defossez, A., Furon, T., &
    Tran, T. (2024). "Proactive Detection of Voice Cloning with Localized
    Watermarking." ICML 2024. https://arxiv.org/abs/2401.17264

A thin adapter over Meta's released `audioseal` package and its pretrained
weights, so the repository can compare its classical methods against a
current neural baseline on the same covers, metrics and attacks.

The pretrained generator carries a fixed 16-bit payload, so longer messages
are split across consecutive chunks of the cover, one payload per chunk. The
package and its weights are an optional dependency: install with
`pip install the-a-files[neural]`.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np

from taf.models.SteganographyMethod import SteganographyMethod

# AudioSeal's generator runs through torch.compile, which shells out to a C++
# compiler. On a machine without one (a stock Windows install, for instance)
# that raises instead of falling back, so the eager path is selected up front.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

_PAYLOAD_BITS = 16


class AudioSealMethod(SteganographyMethod):
    """Neural watermarking with AudioSeal's pretrained models."""

    def __init__(
        self,
        sr: int = 16000,
        generator: str = "audioseal_wm_16bits",
        detector: str = "audioseal_detector_16bits",
        alpha: float = 1.0,
        min_chunk_length: int = 16000,
    ):
        """
        Args:
            sr: Sampling rate of the audio handed to the model.
            generator: Name of the pretrained generator checkpoint.
            detector: Name of the pretrained detector checkpoint.
            alpha: Scale applied to the generated watermark before it is added
                to the carrier. Below 1.0 trades detection for transparency.
            min_chunk_length: Shortest chunk that may carry one payload; sets
                the capacity together with the payload width.
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if min_chunk_length < 1:
            raise ValueError("min_chunk_length must be positive")

        self.sr = sr
        self.generator = generator
        self.detector = detector
        self.alpha = alpha
        self.min_chunk_length = min_chunk_length
        self._models = None

    def _load(self):
        if self._models is None:
            from audioseal import AudioSeal

            self._models = (
                AudioSeal.load_generator(self.generator),
                AudioSeal.load_detector(self.detector),
            )
        return self._models

    def _chunk_bounds(self, sample_count: int, bit_count: int) -> np.ndarray:
        """Chunk edges, one chunk per 16-bit payload."""
        if bit_count <= 0:
            raise ValueError("message must not be empty")

        chunk_count = int(np.ceil(bit_count / _PAYLOAD_BITS))
        if sample_count // chunk_count < self.min_chunk_length:
            capacity = (sample_count // self.min_chunk_length) * _PAYLOAD_BITS
            raise ValueError(
                f"message too long for cover audio: {bit_count} > {capacity} bits"
            )
        return np.linspace(0, sample_count, chunk_count + 1, dtype=np.int64)

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        import torch

        bits = [int(bit) for bit in message]
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError("message must contain only 0 and 1 bits")
        if not bits:
            return data.copy()

        generator, _ = self._load()
        audio = np.asarray(data, dtype=np.float32)
        bounds = self._chunk_bounds(len(audio), len(bits))
        stego = audio.copy()

        for index in range(len(bounds) - 1):
            start, stop = int(bounds[index]), int(bounds[index + 1])
            payload = bits[index * _PAYLOAD_BITS:(index + 1) * _PAYLOAD_BITS]
            # The last chunk may be short of a full payload; pad it, and the
            # decoder simply drops the bits past the message length.
            payload = (payload + [0] * _PAYLOAD_BITS)[:_PAYLOAD_BITS]

            chunk = torch.from_numpy(stego[start:stop])[None, None, :]
            with torch.no_grad():
                watermark = generator.get_watermark(
                    chunk, self.sr, message=torch.tensor(payload, dtype=torch.int32)[None, :]
                )
            stego[start:stop] = (chunk + self.alpha * watermark).squeeze().numpy()

        return stego.astype(data.dtype, copy=False)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        import torch

        if watermark_length <= 0:
            return []

        _, detector = self._load()
        audio = np.asarray(data_with_watermark, dtype=np.float32)
        bounds = self._chunk_bounds(len(audio), watermark_length)

        bits: List[int] = []
        for index in range(len(bounds) - 1):
            start, stop = int(bounds[index]), int(bounds[index + 1])
            chunk = torch.from_numpy(audio[start:stop])[None, None, :]
            with torch.no_grad():
                _, payload = detector.detect_watermark(chunk, self.sr)
            bits.extend(int(bit) for bit in payload[0].tolist())

        return bits[:watermark_length]

    def type(self) -> str:
        return "AudioSeal neural watermarking (pretrained)"
