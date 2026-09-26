"""WavMark neural watermarking.

Reference:
    Chen, G., Wu, Y., Liu, S., Liu, T., Du, X., & Wei, F. (2023). "WavMark:
    Watermarking for Audio Generation." https://arxiv.org/abs/2308.12770

A thin adapter over the released `wavmark` package and its pretrained
weights. WavMark embeds 32 bits per one-second window, of which the first 16
are a fixed synchronisation pattern: the decoder slides that pattern over the
signal to find the windows, which is what lets it recover the payload without
knowing where the watermark starts.

The usable payload is therefore 16 bits per window, and longer messages are
split across consecutive chunks of the cover. The package and its weights are
an optional dependency: install with `pip install the-a-files[neural]`.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np

from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod

# Keep torch on the eager path: see AudioSealMethod for why.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

_PAYLOAD_BITS = 16


class WavMarkMethod(SteganographyMethod):
    """Neural watermarking with WavMark's pretrained model."""

    def __init__(self, sr: int = 16000, min_chunk_length: int = 32000):
        """
        Args:
            sr: Sampling rate the pretrained model expects (16 kHz).
            min_chunk_length: Shortest chunk that may carry one payload. The
                model needs a full one-second window plus room for the
                decoder's pattern search, so two seconds is the practical
                floor.
        """
        if sr != 16000:
            raise ValueError("the pretrained WavMark model is 16 kHz only")
        if min_chunk_length < 16000:
            raise ValueError("min_chunk_length must be at least one second")

        self.sr = sr
        self.min_chunk_length = min_chunk_length
        self._model = None

    def _load(self):
        if self._model is None:
            import wavmark

            self._model = wavmark.load_model()
        return self._model

    def _chunk_bounds(self, sample_count: int, bit_count: int) -> np.ndarray:
        if bit_count <= 0:
            raise ValueError("message must not be empty")

        chunk_count = int(np.ceil(bit_count / _PAYLOAD_BITS))
        if sample_count // chunk_count < self.min_chunk_length:
            capacity = (sample_count // self.min_chunk_length) * _PAYLOAD_BITS
            raise CapacityError(
                f"message too long for cover audio: {bit_count} > {capacity} bits"
            )
        return np.linspace(0, sample_count, chunk_count + 1, dtype=np.int64)

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        import wavmark

        bits = [int(bit) for bit in message]
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError("message must contain only 0 and 1 bits")
        if not bits:
            return data.copy()

        model = self._load()
        audio = np.asarray(data, dtype=np.float32)
        bounds = self._chunk_bounds(len(audio), len(bits))
        stego = audio.copy()

        for index in range(len(bounds) - 1):
            start, stop = int(bounds[index]), int(bounds[index + 1])
            payload = bits[index * _PAYLOAD_BITS:(index + 1) * _PAYLOAD_BITS]
            payload = (payload + [0] * _PAYLOAD_BITS)[:_PAYLOAD_BITS]

            watermarked, _ = wavmark.encode_watermark(
                model, stego[start:stop], np.array(payload), show_progress=False
            )
            stego[start:stop] = watermarked[:stop - start]

        return stego.astype(data.dtype, copy=False)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        import wavmark

        if watermark_length <= 0:
            return []

        model = self._load()
        audio = np.asarray(data_with_watermark, dtype=np.float32)
        bounds = self._chunk_bounds(len(audio), watermark_length)

        bits: List[int] = []
        for index in range(len(bounds) - 1):
            start, stop = int(bounds[index]), int(bounds[index + 1])
            payload, _ = wavmark.decode_watermark(model, audio[start:stop], show_progress=False)

            # The decoder reports None when it cannot find its synchronisation
            # pattern, which is what a destroyed chunk looks like. Report that
            # as zeros rather than a short message, so the caller always gets
            # watermark_length bits and the BER reflects the loss.
            if payload is None:
                bits.extend([0] * _PAYLOAD_BITS)
            else:
                bits.extend(int(bit) for bit in payload)

        return bits[:watermark_length]

    def type(self) -> str:
        return "WavMark neural watermarking (pretrained)"
