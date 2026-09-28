"""Self-synchronising DWT-DCT watermarking with time-domain sync codes.

Reference:
    Wang, X.-Y., & Zhao, H. (2006). "A novel synchronization invariant audio
    watermarking scheme based on DWT and DCT." IEEE Transactions on Signal
    Processing, 54(12), 4835-4840.
    https://doi.org/10.1109/TSP.2006.881258

The cover is cut into blocks of a synchronisation code followed by a data
segment:

* The 16-bit sync code (the 13-bit Barker code followed by the 3-bit one) is
  written into the time domain, one bit per group of consecutive samples, by
  quantising the group mean.
* The data segment is decomposed by a multi-level DWT, its approximation band
  is transformed by a DCT, and each payload bit quantises one low-frequency
  DCT coefficient.

The decoder does not assume where blocks start. It slides over the signal,
scores every offset by how closely the group means sit on the sync code's
quantisers, confirms a candidate by how closely the following data
coefficients sit on either quantiser, and reads the payload from every
confirmed block. The mark is therefore found again after the signal has been
cropped, shifted or padded - desynchronisation that defeats schemes reading a
fixed frame grid.

Which coefficients carry the payload and how the steps are sized are this
implementation's choices; the paper's own rules were not available to it.
Both quantisation steps are relative, as absolute strengths are not portable
across recording levels: the sync step to the RMS of the data
segment that follows it, and the data step to the RMS of the DCT coefficients
that carry no payload. Neither reference is disturbed by the quantisation it
scales, so the decoder recomputes the steps the embedder used.

A message longer than one block is split into consecutive chunks; blocks
repeat the chunks cyclically and the decoder votes over the repetitions.
Chunks are numbered from the first block found, so cutting away the start of
the signal only preserves a message that fits in one block.
"""
from typing import Dict, List, Tuple

import numpy as np
import pywt
from scipy.fft import dct, idct

from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod


class SyncDwtDctMethod(SteganographyMethod):
    """Sync-code DWT-DCT watermarking (Wang & Zhao)."""

    #: 13-bit Barker code followed by the 3-bit Barker code.
    SYNC_CODE = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0])
    #: Lower bound on either step, to keep digital silence well defined.
    MIN_STEP = 1e-7
    #: Candidate offsets whose sync score exceeds this are not examined.
    SYNC_GATE = 0.3

    def __init__(self, sr: int = 16000, segment_length: int = 4096, sync_group: int = 16,
                 level: int = 3, wavelet: str = "db4", band_start: float = 200.0,
                 bits_per_block: int = 32, step_scale: float = 0.5, sync_scale: float = 0.3,
                 detection_threshold: float = 0.2):
        """
        Args:
            sr: Sampling rate, used to place the carrier band.
            segment_length: Samples in the data part of each block; must be a
                multiple of ``2 ** level``.
            sync_group: Consecutive samples whose mean carries one sync bit.
                The paper uses 5; longer groups are more robust to noise.
            level: DWT decomposition depth.
            wavelet: PyWavelets wavelet name.
            band_start: Frequency in Hz of the first carrier coefficient.
            bits_per_block: Most payload bits one data segment carries.
            step_scale: Data quantisation step, relative to the RMS of the
                approximation-band DCT coefficients that carry no payload.
            sync_scale: Sync quantisation step, relative to the RMS of the
                data segment that follows the code.
            detection_threshold: Largest mean normalised quantisation residual
                (0 = on the lattice, 0.5 = random) accepted as a block.
        """
        if level < 1:
            raise ValueError("level must be at least 1")
        if segment_length <= 0 or segment_length % (2 ** level):
            raise ValueError("segment_length must be a positive multiple of 2 ** level")
        if sync_group < 1:
            raise ValueError("sync_group must be at least 1")
        if bits_per_block < 1:
            raise ValueError("bits_per_block must be at least 1")
        if step_scale <= 0 or sync_scale <= 0:
            raise ValueError("step_scale and sync_scale must be positive")
        if not 0 < detection_threshold < 0.5:
            raise ValueError("detection_threshold must be between 0 and 0.5")

        self.sr = sr
        self.segment_length = segment_length
        self.sync_group = sync_group
        self.level = level
        self.wavelet = wavelet
        self.band_start = band_start
        self.bits_per_block = bits_per_block
        self.step_scale = step_scale
        self.sync_scale = sync_scale
        self.detection_threshold = detection_threshold

        self.sync_length = len(self.SYNC_CODE) * sync_group
        self.block_length = self.sync_length + segment_length

        approximation_length = segment_length // 2 ** level
        approximation_rate = sr / 2 ** level
        # DCT-II coefficient k of an M-point band sampled at fs sits at k * fs / (2M).
        self.first_coefficient = int(round(band_start * 2 * approximation_length / approximation_rate))
        if self.first_coefficient < 1 or self.first_coefficient + bits_per_block > approximation_length:
            raise ValueError(
                "band_start and bits_per_block place the carrier outside the approximation band"
            )

    # -- layout -------------------------------------------------------------

    def _layout(self, sample_count: int, bit_count: int) -> Tuple[int, int, int]:
        """Bits per block, number of chunks and number of blocks."""
        if bit_count <= 0:
            raise ValueError("message must not be empty")

        block_count = sample_count // self.block_length
        per_block = min(bit_count, self.bits_per_block)
        chunk_count = -(-bit_count // per_block)
        if chunk_count > block_count:
            raise CapacityError(
                f"message too long for cover audio: {bit_count} > "
                f"{block_count * self.bits_per_block} bits"
            )
        return per_block, chunk_count, block_count

    @staticmethod
    def _chunk(message: List[int], chunk: int, per_block: int) -> List[int]:
        return list(message[chunk * per_block:(chunk + 1) * per_block])

    # -- quantisation ---------------------------------------------------------

    @staticmethod
    def _quantize(value: np.ndarray, step: np.ndarray, bit: np.ndarray) -> np.ndarray:
        dither = np.where(bit == 1, step / 4.0, -step / 4.0)
        return np.round((value - dither) / step) * step + dither

    @staticmethod
    def _residual(value: np.ndarray, step: np.ndarray, bit: np.ndarray) -> np.ndarray:
        """Distance to the bit's quantiser, 0 on the lattice and 1 half a step off."""
        dither = np.where(bit == 1, step / 4.0, -step / 4.0)
        position = (value - dither) / step
        return 2.0 * np.abs(position - np.round(position))

    @staticmethod
    def _either_residual(value: np.ndarray, step: np.ndarray) -> np.ndarray:
        """Distance to the nearer of both quantisers, 0 on one and 1 midway."""
        position = (value + step / 4.0) / (step / 2.0)
        return 2.0 * np.abs(position - np.round(position))

    # -- data segment -------------------------------------------------------

    def _band(self, segment: np.ndarray, bit_count: int):
        coefficients = pywt.wavedec(segment, self.wavelet, mode="periodization", level=self.level)
        spectrum = dct(coefficients[0], norm="ortho")
        carrier = np.arange(self.first_coefficient, self.first_coefficient + bit_count)

        # Coefficients that carry no payload are left untouched, so their RMS
        # is the same before and after embedding.
        rest = np.delete(spectrum, carrier)
        step = max(self.step_scale * float(np.sqrt(np.mean(rest ** 2))), self.MIN_STEP)
        return coefficients, spectrum, carrier, step

    def _embed_segment(self, segment: np.ndarray, bits: List[int]) -> np.ndarray:
        coefficients, spectrum, carrier, step = self._band(segment, len(bits))
        spectrum[carrier] = self._quantize(spectrum[carrier], np.full(len(bits), step), np.asarray(bits))
        coefficients[0] = idct(spectrum, norm="ortho")
        return pywt.waverec(coefficients, self.wavelet, mode="periodization")[:len(segment)]

    def _read_segment(self, segment: np.ndarray, bit_count: int) -> Tuple[List[int], float]:
        """Decoded bits and the mean residual to the nearer quantiser."""
        _, spectrum, carrier, step = self._band(segment, bit_count)
        values = spectrum[carrier]
        steps = np.full(bit_count, step)
        residual_zero = self._residual(values, steps, np.zeros(bit_count, dtype=int))
        residual_one = self._residual(values, steps, np.ones(bit_count, dtype=int))
        bits = [int(one < zero) for zero, one in zip(residual_zero, residual_one)]
        return bits, float(np.mean(self._either_residual(values, steps)))

    # -- sync code ------------------------------------------------------------

    def _embed_sync(self, stego: np.ndarray, start: int) -> None:
        data = stego[start + self.sync_length:start + self.block_length]
        step = max(self.sync_scale * float(np.sqrt(np.mean(data ** 2))), self.MIN_STEP)

        groups = stego[start:start + self.sync_length].reshape(len(self.SYNC_CODE), self.sync_group)
        means = groups.mean(axis=1)
        targets = self._quantize(means, np.full(len(means), step), self.SYNC_CODE)
        groups += (targets - means)[:, None]
        stego[start:start + self.sync_length] = groups.reshape(-1)

    def _sync_scores(self, audio: np.ndarray) -> np.ndarray:
        """Mean sync residual at every offset where a whole block fits."""
        offsets = len(audio) - self.block_length + 1
        if offsets <= 0:
            return np.empty(0)

        cumulative = np.concatenate(([0.0], np.cumsum(audio)))
        energy = np.concatenate(([0.0], np.cumsum(audio ** 2)))
        positions = np.arange(offsets)

        data_start = positions + self.sync_length
        data_energy = energy[data_start + self.segment_length] - energy[data_start]
        steps = np.maximum(self.sync_scale * np.sqrt(np.maximum(data_energy, 0.0) / self.segment_length),
                           self.MIN_STEP)

        scores = np.zeros(offsets)
        for index, bit in enumerate(self.SYNC_CODE):
            start = positions + index * self.sync_group
            means = (cumulative[start + self.sync_group] - cumulative[start]) / self.sync_group
            scores += self._residual(means, steps, np.full(offsets, bit))
        return scores / len(self.SYNC_CODE)

    def _find_blocks(self, audio: np.ndarray, per_block: int) -> List[Tuple[int, List[int]]]:
        """Confirmed block positions with the bits read from each."""
        scores = self._sync_scores(audio)
        candidates = np.where(scores < self.SYNC_GATE)[0]
        candidates = candidates[np.argsort(scores[candidates], kind="stable")]

        found: List[Tuple[int, List[int]]] = []
        taken: List[int] = []
        for position in candidates:
            if any(abs(position - other) < self.block_length // 2 for other in taken):
                continue
            data_start = position + self.sync_length
            bits, data_score = self._read_segment(audio[data_start:data_start + self.segment_length], per_block)
            sync_count = len(self.SYNC_CODE)
            combined = (scores[position] * sync_count + data_score * per_block) / (sync_count + per_block)
            if combined < self.detection_threshold:
                found.append((int(position), bits))
                taken.append(int(position))
        return sorted(found)

    # -- interface ------------------------------------------------------------

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        audio = np.asarray(data, dtype=np.float64)
        per_block, chunk_count, block_count = self._layout(len(audio), len(message))
        stego = audio.copy()

        for block in range(block_count):
            start = block * self.block_length
            data_start = start + self.sync_length
            bits = self._chunk(message, block % chunk_count, per_block)
            if len(bits) < per_block:
                # The last chunk is padded so every block has the same layout.
                bits = bits + [0] * (per_block - len(bits))
            stego[data_start:data_start + self.segment_length] = self._embed_segment(
                stego[data_start:data_start + self.segment_length], bits
            )
            # The sync step depends on the marked data segment, so the code
            # goes in second.
            self._embed_sync(stego, start)

        return stego.astype(np.asarray(data).dtype, copy=False)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        audio = np.asarray(data_with_watermark, dtype=np.float64)
        if watermark_length <= 0:
            raise ValueError("message must not be empty")
        per_block = min(watermark_length, self.bits_per_block)
        chunk_count = -(-watermark_length // per_block)

        blocks = self._find_blocks(audio, per_block)
        if not blocks:
            # Nothing passed detection: fall back to the embedding grid.
            blocks = []
            for block in range(len(audio) // self.block_length):
                data_start = block * self.block_length + self.sync_length
                bits, _ = self._read_segment(audio[data_start:data_start + self.segment_length], per_block)
                blocks.append((block * self.block_length, bits))

        votes: Dict[int, np.ndarray] = {}
        origin = blocks[0][0] if blocks else 0
        for position, bits in blocks:
            chunk = int(round((position - origin) / self.block_length)) % chunk_count
            tally = votes.setdefault(chunk, np.zeros(per_block))
            tally += np.where(np.asarray(bits) == 1, 1.0, -1.0)

        message: List[int] = []
        for chunk in range(chunk_count):
            tally = votes.get(chunk, np.zeros(per_block))
            message.extend(int(value > 0) for value in tally)
        return message[:watermark_length]

    def type(self) -> str:
        return "Synchronisation-code DWT-DCT watermarking (Sync-DWT-DCT)"
