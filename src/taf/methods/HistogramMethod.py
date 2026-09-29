"""Histogram-based audio watermarking.

Reference:
    Xiang, S., & Huang, J. (2007). "Histogram-Based Audio Watermarking Against
    Time-Scale Modification and Cropping Attacks." IEEE Transactions on
    Multimedia, 9(7), 1357-1372. https://doi.org/10.1109/TMM.2007.906580

Every other method in this repository reads its payload from fixed sample
positions, so stretching the time axis or cutting a piece out desynchronises
the decoder and the message is lost. The amplitude histogram does not depend
on sample order at all: resampling, time-scale modification and cropping
leave its shape nearly intact.

A bit is carried by the population relation inside a group of three
neighbouring histogram bins, and the amplitude range the histogram covers is
expressed as a multiple of the mean absolute amplitude, which also makes the
embedding invariant to volume changes.
"""
from typing import List, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.card import MethodCard, Reference, Text


class HistogramMethod(SteganographyMethod):
    """Watermarking in the amplitude histogram."""

    card = MethodCard(
        title="Histogram",
        family="statistical",
        purpose="watermarking",
        strength_parameter="threshold",
        references=(Reference("Xiang & Huang", 2007, doi="10.1109/TMM.2007.906580"),),
        summary=Text(
            en="Stores bits in the distribution of sample amplitudes rather than their positions.",
            pl="Zapisuje bity w rozkładzie amplitud próbek zamiast w ich pozycjach.",
        ),
        details=Text(
            en=(
                "Forms an amplitude histogram and modifies population relations in groups of three "
                "neighbouring bins. The histogram range is relative to mean absolute amplitude; "
                "threshold controls the required relation. Because sample order is not used, the method "
                "targets timing and cropping robustness. Changes that reshape amplitude statistics, "
                "such as clipping or noise, can still destroy the relation."
            ),
            pl=(
                "Buduje histogram amplitud i zmienia relacje liczności w grupach trzech sąsiednich "
                "przedziałów. Zakres histogramu zależy od średniej amplitudy bezwzględnej, a threshold "
                "określa wymaganą relację. Brak zależności od kolejności próbek służy odporności na "
                "zmiany czasowe i przycięcie. Zmiany statystyk amplitudy, np. clipping lub szum, mogą "
                "jednak zniszczyć tę relację."
            ),
        ),
    )

    def __init__(self, sr: int = 16000, amplitude_span: float = 2.5, threshold: float = 2.0,
                 cutoff: float = 2000.0, rounds: int = 24,
                 search_span: float = 0.2, search_steps: int = 41,
                 min_samples_per_bin: int = 32):
        """
        Args:
            sr: Sampling rate, needed for the low-frequency split.
            amplitude_span: Half-width of the embedded amplitude range, as a
                multiple of the mean absolute amplitude.
            threshold: Population ratio the embedder enforces between the
                middle bin and its two neighbours. The decoder decides at 1,
                so the excess is the margin an attack has to eat through.
            search_span: Relative range of bin-edge scale factors the
                decoder searches over, to re-align the histogram after an
                attack has changed the mean absolute amplitude.
            search_steps: Number of scale factors tried within that range.
            min_samples_per_bin: Samples a histogram bin needs before it can
                carry a relation; sets the capacity together with the bit
                count.
            rounds: Maximum embedding rounds. Each round re-counts the
                populations in the band-limited signal the decoder will see.
            cutoff: Upper edge of the low-frequency component that carries the
                histogram, in Hz. Embedding in the full-band signal leaves the
                relation at the mercy of high-frequency detail, which noise,
                filtering and resampling all disturb.
        """
        if amplitude_span <= 0:
            raise ValueError("amplitude_span must be positive")
        if threshold <= 1:
            raise ValueError("threshold must be greater than 1")
        if not 0 < cutoff < sr / 2:
            raise ValueError("cutoff must be between 0 and the Nyquist frequency")

        self.sr = sr
        self.amplitude_span = amplitude_span
        self.threshold = threshold
        self.cutoff = cutoff
        self.rounds = rounds
        self.search_span = search_span
        self.search_steps = search_steps
        self.min_samples_per_bin = min_samples_per_bin

    def _split(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split into the low-frequency carrier and the untouched remainder."""
        sos = butter(4, self.cutoff, fs=self.sr, btype="low", output="sos")
        low = sosfiltfilt(sos, audio)
        return low, audio - low

    def _bin_edges(self, audio: np.ndarray, bit_count: int) -> Tuple[np.ndarray, float]:
        """Histogram edges, scaled by the mean absolute amplitude."""
        mean_amplitude = float(np.mean(np.abs(audio)))
        if mean_amplitude == 0.0:
            raise ValueError("cover audio is silent")

        # Three bins per bit, and a bin with almost no samples in it cannot
        # carry a population relation at all.
        capacity = len(audio) // (3 * self.min_samples_per_bin)
        if bit_count > capacity:
            raise CapacityError(
                f"message too long for cover audio: {bit_count} > {capacity} bits"
            )

        upper = self.amplitude_span * mean_amplitude
        bin_count = 3 * bit_count
        edges = np.linspace(0.0, upper, bin_count + 1)
        return edges, float(edges[1] - edges[0])

    @staticmethod
    def _bin_index(magnitudes: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Bin of every sample; -1 for samples outside the embedded range."""
        index = np.digitize(magnitudes, edges) - 1
        index[(magnitudes >= edges[-1]) | (magnitudes < edges[0])] = -1
        return index

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        audio = np.asarray(data, dtype=np.float64)
        if len(message) == 0:
            return data.copy()

        low, high_band = self._split(audio)
        edges, width = self._bin_edges(low, len(message))
        stego = low.copy()

        # Moving a sample by a fraction of a bin is a broadband change, so the
        # modified signal is no longer purely low-frequency. The decoder only
        # ever sees the low-passed version, so each round is projected back
        # into the band and the populations are re-counted there; without the
        # projection the relation holds in a signal nobody reads, and the
        # margin the decoder actually sees falls short of the threshold.
        for _ in range(self.rounds):
            index = self._bin_index(np.abs(stego), edges)
            satisfied = True

            for group, bit in enumerate(message):
                lower_bin, middle_bin, upper_bin = 3 * group, 3 * group + 1, 3 * group + 2
                counts = [int(np.count_nonzero(index == b))
                          for b in (lower_bin, middle_bin, upper_bin)]
                ratio = self._ratio(counts)

                if (int(bit) == 1 and ratio >= self.threshold) or (
                    int(bit) == 0 and ratio <= 1.0 / self.threshold
                ):
                    continue

                satisfied = False
                if int(bit) == 1:
                    # Needs a fuller middle bin: pull samples in from the
                    # neighbour that can spare them.
                    source = lower_bin if counts[0] >= counts[2] else upper_bin
                    moved = self._samples_to_move(counts, self.threshold, into_middle=True)
                    self._move(stego, index, source, middle_bin, edges, width, moved)
                else:
                    # Needs an emptier middle bin: push its samples outward.
                    target = lower_bin if counts[0] <= counts[2] else upper_bin
                    moved = self._samples_to_move(counts, self.threshold, into_middle=False)
                    self._move(stego, index, middle_bin, target, edges, width, moved)

            if satisfied:
                break

            # Band-limit only the modification, not the carrier: re-filtering
            # the whole signal every round compounds the filter's roll-off and
            # costs a fixed ~26 dB of SNR no matter how little is embedded.
            modification, _ = self._split(stego - low)
            stego = low + modification
            edges, width = self._bin_edges(stego, len(message))

        return (stego + high_band).astype(data.dtype, copy=False)

    @staticmethod
    def _ratio(counts: List[int]) -> float:
        """2*h_middle / (h_low + h_high), the paper's population relation."""
        outer = counts[0] + counts[2]
        if outer == 0:
            return float("inf") if counts[1] > 0 else 1.0
        return 2.0 * counts[1] / outer

    @staticmethod
    def _samples_to_move(counts: List[int], threshold: float, into_middle: bool) -> int:
        """How many samples must change bin to reach the requested relation.

        Solved directly rather than nudged a fraction at a time: every sample
        moved both fills one bin and empties another, so an iterative nudge
        converges slowly and used to stop at the iteration cap with the
        relation still short of the threshold, leaving groups with almost no
        margin for an attack to eat through.
        """
        middle, outer = counts[1], counts[0] + counts[2]
        if into_middle:
            # (2*(middle + k)) / (outer - k) >= threshold
            needed = (threshold * outer - 2 * middle) / (threshold + 2)
        else:
            # (2*(middle - k)) / (outer + k) <= 1 / threshold
            needed = (2 * threshold * middle - outer) / (2 * threshold + 1)
        return max(1, int(np.ceil(needed)))

    def _move(self, stego: np.ndarray, index: np.ndarray, source: int, target: int,
              edges: np.ndarray, width: float, count: int) -> None:
        """Shift a few samples from one bin into an adjacent one.

        The samples nearest the shared boundary are moved first, so each one
        travels the shortest distance that changes its bin.
        """
        candidates = np.where(index == source)[0]
        if candidates.size == 0:
            return

        boundary = edges[target] if target > source else edges[target + 1]
        distance = np.abs(np.abs(stego[candidates]) - boundary)
        order = candidates[np.argsort(distance)]

        batch = order[:max(1, min(count, len(order)))]
        direction = 1.0 if target > source else -1.0
        magnitudes = np.abs(stego[batch]) + direction * width
        lower, upper = edges[target], edges[target + 1]
        # Land strictly inside the target bin.
        magnitudes = np.clip(magnitudes, lower + width * 0.01, upper - width * 0.01)
        stego[batch] = np.sign(stego[batch]) * magnitudes

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        audio = np.asarray(data_with_watermark, dtype=np.float64)
        if watermark_length == 0:
            return []

        low, _ = self._split(audio)
        edges, _ = self._bin_edges(low, watermark_length)
        magnitudes = np.abs(low)

        # Cropping or time-scaling shifts the mean absolute amplitude that
        # sets the bin edges, which slides every sample a little way along the
        # histogram. Searching a small range of scale factors and keeping the
        # one whose population relations come out most decisive recovers the
        # alignment, as the paper's exhaustive search does.
        best_bits: List[int] = []
        best_confidence = -np.inf

        for scale in np.linspace(1.0 - self.search_span, 1.0 + self.search_span,
                                 self.search_steps):
            index = self._bin_index(magnitudes, edges * scale)
            bits: List[int] = []
            confidence = 0.0

            for group in range(watermark_length):
                counts = [int(np.count_nonzero(index == 3 * group + offset))
                          for offset in range(3)]
                ratio = self._ratio(counts)
                bits.append(1 if ratio >= 1.0 else 0)
                if np.isfinite(ratio) and ratio > 0:
                    confidence += abs(np.log(ratio))

            if confidence > best_confidence:
                best_confidence = confidence
                best_bits = bits

        return best_bits

    def type(self) -> str:
        return "Histogram-based method robust to time-scale modification"
