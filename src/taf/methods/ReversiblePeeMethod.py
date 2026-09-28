"""Reversible data hiding by prediction-error expansion (PEE).

References:
    Thodi, D. M., & Rodriguez, J. J. (2007). "Expansion embedding techniques
    for reversible watermarking." IEEE Transactions on Image Processing,
    16(3), 721-730. https://doi.org/10.1109/TIP.2006.891046

    Nishimura, A. (2011). "Reversible audio data hiding using linear
    prediction and error expansion." Proceedings of IIHMSP 2011, 318-321.
    https://doi.org/10.1109/IIHMSP.2011.76

Reversible (lossless) data hiding returns not only the payload but the exact
original cover. Samples are handled as 16-bit PCM integers. Each sample is
predicted from the two samples before it, ``p = 2*y[n-1] - y[n-2]``, and the
prediction error ``e = x - p`` is either

* expanded, ``e' = 2e + bit``, when ``-T <= e < T``, which stores one bit; or
* shifted by ``T`` away from zero otherwise, which vacates the range used by
  expanded errors so the decoder can tell the two cases apart.

The decoder inverts both mappings: ``e = floor(e' / 2)`` with bit ``e' mod 2``
inside ``[-2T, 2T)``, and ``e = e' -/+ T`` outside.

The prediction uses the *original* previous samples. The decoder runs forward
and has restored them by the time it needs them; predicting from the marked
samples instead would spare it that order but costs much of the capacity, as
the embedding noise enters every prediction. Samples within ``T`` of full scale
could overflow and are skipped; their positions travel in a location map in
the least significant bits of the first samples, whose original bits are
appended to the payload. Once the payload is stored the remaining samples are
left untouched, so distortion is confined to the prefix the payload needs.

The scheme is fragile by design: any change to the signal, including
conversion to another bit depth, destroys both payload and reversibility.
The payload survives only an in-memory or 16-bit lossless round trip. A cover
not on the 16-bit grid is rounded onto it first, and it is that rounded cover
that ``recover_cover`` returns.
"""
from typing import List, Tuple

import numpy as np

from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod


class ReversiblePeeMethod(SteganographyMethod):
    """Prediction-error expansion reversible data hiding."""

    SCALE = 32768
    LOWEST = -32768
    HIGHEST = 32767
    #: Width of the location-map length field.
    COUNT_BITS = 32

    def __init__(self, threshold: int = 8):
        """
        Args:
            threshold: Expansion threshold ``T``. Errors in ``[-T, T)`` carry
                a bit; the others are shifted by ``T``. Larger values raise
                capacity and distortion.
        """
        if threshold < 1:
            raise ValueError("threshold must be at least 1")
        self.threshold = int(threshold)

    # -- sample conversion ------------------------------------------------------

    def _to_pcm(self, data: np.ndarray) -> np.ndarray:
        audio = np.asarray(data, dtype=np.float64)
        return np.clip(np.round(audio * self.SCALE), self.LOWEST, self.HIGHEST).astype(np.int64)

    def _from_pcm(self, pcm: np.ndarray, dtype: np.dtype) -> np.ndarray:
        return (pcm.astype(np.float64) / self.SCALE).astype(dtype)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _index_bits(sample_count: int) -> int:
        return max(1, int(sample_count - 1).bit_length())

    @staticmethod
    def _to_bits(value: int, width: int) -> List[int]:
        return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]

    @staticmethod
    def _from_bits(bits: List[int]) -> int:
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return value

    def _predict(self, pcm: np.ndarray, index: int) -> int:
        previous = int(pcm[index - 1])
        prediction = 2 * previous - int(pcm[index - 2])
        return min(max(prediction, self.LOWEST), self.HIGHEST)

    # -- embedding ---------------------------------------------------------------

    def _embed(self, pcm: np.ndarray, payload: List[int], skipped: List[int]) -> Tuple[np.ndarray, int]:
        """Mark ``pcm`` with ``payload`` behind a location map of ``skipped``.

        Returns the marked samples and the index after the last one touched.
        """
        index_bits = self._index_bits(len(pcm))
        header = self._to_bits(len(skipped), self.COUNT_BITS)
        for position in skipped:
            header += self._to_bits(position, index_bits)

        header_length = len(header)
        if header_length + 2 > len(pcm):
            raise CapacityError("cover audio too short for the reversible header")

        marked = pcm.copy()
        original_lsbs = [int(value) & 1 for value in pcm[:header_length]]
        marked[:header_length] = (pcm[:header_length] & ~1) | np.asarray(header, dtype=np.int64)
        # What the decoder holds when it reaches each sample: the header as
        # marked (its original bits come last) and every later sample
        # already restored.
        reference = marked.copy()

        stream = list(payload) + original_lsbs
        skip = set(skipped)
        threshold = self.threshold
        written = 0
        index = max(header_length, 2)

        while written < len(stream):
            if index >= len(pcm):
                raise CapacityError(
                    f"message too long for cover audio: needs {len(stream)} expandable "
                    f"samples after the header, found {written}"
                )
            if index not in skip:
                prediction = self._predict(reference, index)
                error = int(pcm[index]) - prediction
                if -threshold <= error < threshold:
                    error = 2 * error + stream[written]
                    written += 1
                elif error >= threshold:
                    error += threshold
                else:
                    error -= threshold
                marked[index] = prediction + error
            index += 1

        return marked, index

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        if len(message) == 0:
            raise ValueError("message must not be empty")
        pcm = self._to_pcm(data)
        if len(message) > len(pcm):
            raise CapacityError(
                f"message too long for cover audio: {len(message)} > {len(pcm)} bits"
            )

        # The location map must list every sample near full scale between the
        # header and the last sample touched. That end point moves with the
        # map's own length, so the map grows until it covers it.
        near = np.where((pcm > self.HIGHEST - self.threshold) | (pcm < self.LOWEST + self.threshold))[0]
        payload = [int(bit) for bit in message]
        skipped: List[int] = []
        while True:
            marked, end = self._embed(pcm, payload, skipped)
            header_length = self.COUNT_BITS + len(skipped) * self._index_bits(len(pcm))
            listed = set(skipped)
            missing = [int(index) for index in near if header_length <= index < end and index not in listed]
            if not missing:
                break
            skipped = sorted(listed.union(missing))

        return self._from_pcm(marked, np.asarray(data).dtype)

    # -- extraction ---------------------------------------------------------------

    def _extract(self, data: np.ndarray, watermark_length: int, strict: bool) -> Tuple[List[int], np.ndarray]:
        """The payload and the restored 16-bit cover.

        With ``strict`` unset, a header that cannot be valid - what any
        processing of the marked signal leaves behind - is read as an empty
        location map, so a damaged signal yields wrong bits rather than an
        exception, like every other method.
        """
        marked = self._to_pcm(data)
        sample_count = len(marked)
        if sample_count < self.COUNT_BITS + 2:
            raise CapacityError("audio too short to hold a reversible header")

        index_bits = self._index_bits(sample_count)
        lsbs = (marked & 1).tolist()
        count = self._from_bits(lsbs[:self.COUNT_BITS])
        if self.COUNT_BITS + count * index_bits + 2 > sample_count:
            if strict:
                raise ValueError("location map longer than the audio; not a marked signal")
            count = 0
        header_length = self.COUNT_BITS + count * index_bits
        skip = {
            self._from_bits(lsbs[self.COUNT_BITS + k * index_bits:self.COUNT_BITS + (k + 1) * index_bits])
            for k in range(count)
        }

        restored = marked.copy()
        stream: List[int] = []
        total = watermark_length + header_length
        threshold = self.threshold
        index = max(header_length, 2)

        while len(stream) < total and index < sample_count:
            if index not in skip:
                prediction = self._predict(restored, index)
                error = int(marked[index]) - prediction
                if -2 * threshold <= error < 2 * threshold:
                    stream.append(error & 1)
                    error >>= 1
                elif error >= 2 * threshold:
                    error -= threshold
                else:
                    error += threshold
                restored[index] = prediction + error
            index += 1

        stream += [0] * (total - len(stream))
        payload = stream[:watermark_length]
        original_lsbs = np.asarray(stream[watermark_length:total], dtype=np.int64)
        restored[:header_length] = (restored[:header_length] & ~1) | original_lsbs
        return payload, restored

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        if watermark_length <= 0:
            raise ValueError("message must not be empty")
        payload, _ = self._extract(data_with_watermark, watermark_length, strict=False)
        return payload

    def recover_cover(self, data_with_watermark: np.ndarray, watermark_length: int) -> np.ndarray:
        """Undo the embedding and return the original cover, sample for sample.

        Exact only for an unmodified marked signal; a cover that was not on
        the 16-bit grid comes back rounded onto it.
        """
        _, restored = self._extract(data_with_watermark, watermark_length, strict=True)
        return self._from_pcm(restored, np.asarray(data_with_watermark).dtype)

    def type(self) -> str:
        return "Reversible prediction-error expansion (PEE)"
