"""Audio watermarking in the extrema of low-frequency EMD modes.

Reference:
    Khaldi, K., & Boudraa, A.-O. (2013). "Audio watermarking via EMD." IEEE
    Transactions on Audio, Speech, and Language Processing, 21(3), 675-680.
    https://doi.org/10.1109/TASL.2012.2227733

Each frame is decomposed by empirical mode decomposition into intrinsic mode
functions (IMFs), and the payload is carried by the extrema of the slowest,
low-frequency component. Here every extremum magnitude ``v`` is quantised to
the nearest point of ``k * S + S/4`` for a zero or ``k * S + 3S/4`` for a one,
and read back by asking which half of its quantisation cell it falls into.

EMD is data-driven rather than a fixed transform, which is the point of the
scheme, but it has no exact inverse: decomposing the marked frame again does
not return the modified mode. This implementation departs from the paper in
three places to obtain a blind, bit-exact decoder:

* The carrier is the remainder after ``imf_count`` IMFs (the last IMFs plus
  the residue), not the last IMF alone. Energy moving between the two slowest
  modes then no longer changes the carrier.
* Embedding is closed-loop: the marked frame is decomposed again and
  re-marked until the decoder's view agrees with the bit.
* Each frame carries a single bit, voted on by all of its carrier extrema,
  and the message repeats over the frames. The number of extrema is
  therefore free to change after decomposition. The paper's synchronisation
  code is not reproduced; see ``SyncDwtDctMethod`` for a sync-code scheme.

The quantisation step is relative to the RMS of the removed, higher-frequency
IMFs, which marking leaves nearly untouched, so it follows the signal level
and survives a volume change.

What the carrier contains depends on the decomposition, and processing that
removes or adds oscillations - low-pass filtering above all, but also noise
and lossy coding - changes which content ends up in it. This is a property
of EMD rather than of the quantiser, and it makes the scheme markedly less
robust to filtering than transform-domain methods with a fixed basis.
"""
from typing import List, Tuple

import numpy as np

from taf.methods.common.emd import emd, extrema
from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod


class EmdMethod(SteganographyMethod):
    """EMD extrema quantisation (Khaldi & Boudraa)."""

    #: Share of a frame's carrier extrema that must read back correctly
    #: before closed-loop embedding stops refining it.
    AGREEMENT = 0.75

    def __init__(self, frame_length: int = 1024, imf_count: int = 4, step_scale: float = 0.2,
                 sift_iterations: int = 4, edge_margin: int = 64, max_refinements: int = 6):
        """
        Args:
            frame_length: Samples per frame; each frame carries one bit.
            imf_count: IMFs removed before the remainder that carries the
                payload. More IMFs leave a slower carrier with fewer extrema.
            step_scale: Quantisation step, relative to the RMS of the removed
                IMFs. Larger steps are more robust and more audible.
            sift_iterations: Fixed number of sifting passes per IMF.
            edge_margin: Extrema this close to a frame edge are ignored; the
                spline envelopes are least reliable there.
            max_refinements: Upper bound on closed-loop embedding passes.
        """
        if frame_length < 64:
            raise ValueError("frame_length must be at least 64")
        if imf_count < 1:
            raise ValueError("imf_count must be at least 1")
        if step_scale <= 0:
            raise ValueError("step_scale must be positive")
        if sift_iterations < 1:
            raise ValueError("sift_iterations must be at least 1")
        if not 0 <= edge_margin < frame_length // 2:
            raise ValueError("edge_margin must be below half the frame length")
        if max_refinements < 1:
            raise ValueError("max_refinements must be at least 1")

        self.frame_length = frame_length
        self.imf_count = imf_count
        self.step_scale = step_scale
        self.sift_iterations = sift_iterations
        self.edge_margin = edge_margin
        self.max_refinements = max_refinements

    def _frame_count(self, sample_count: int, bit_count: int) -> int:
        if bit_count <= 0:
            raise ValueError("message must not be empty")

        frame_count = sample_count // self.frame_length
        if bit_count > frame_count:
            raise CapacityError(
                f"message too long for cover audio: {bit_count} > {frame_count} bits"
            )
        return frame_count

    def _analyse(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Split a frame into its fast IMFs and the carrier, with the step and
        the usable carrier extrema."""
        imfs, _ = emd(frame, self.imf_count, self.sift_iterations)
        fast = np.sum(imfs, axis=0) if imfs else np.zeros_like(frame)
        carrier = frame - fast

        step = self.step_scale * float(np.sqrt(np.mean(fast ** 2)))
        if step == 0.0:
            step = self.step_scale * float(np.sqrt(np.mean(frame ** 2))) or 1e-9

        maxima, minima = extrema(carrier)
        points = np.sort(np.concatenate((maxima, minima)))
        return fast, carrier, step, points

    def _usable(self, points: np.ndarray) -> np.ndarray:
        return points[(points >= self.edge_margin) & (points < self.frame_length - self.edge_margin)]

    @staticmethod
    def _fraction(values: np.ndarray, step: float) -> np.ndarray:
        """Position of each extremum magnitude inside its quantisation cell."""
        magnitudes = np.abs(values)
        return (magnitudes - np.floor(magnitudes / step) * step) / step

    def _votes(self, frame: np.ndarray) -> np.ndarray:
        """One vote per usable carrier extremum: +1 for a one, -1 for a zero."""
        _, carrier, step, points = self._analyse(frame)
        usable = self._usable(points)
        return np.where(self._fraction(carrier[usable], step) >= 0.5, 1.0, -1.0)

    def _mark(self, frame: np.ndarray, bit: int) -> np.ndarray:
        fast, carrier, step, points = self._analyse(frame)
        marked = carrier.copy()
        dither = 0.75 * step if bit else 0.25 * step

        for position in self._usable(points):
            magnitude = abs(carrier[position])
            target = np.round((magnitude - dither) / step) * step + dither
            if target < 0:
                target += step

            # Raise or lower the extremum with a raised-cosine bump that ends
            # at the neighbouring extrema: moving the single sample would
            # create a spike, and the next decomposition would treat it as a
            # new, fast oscillation.
            where = np.searchsorted(points, position)
            start = points[where - 1] if where > 0 else max(0, position - self.edge_margin)
            stop = points[where + 1] if where + 1 < len(points) else min(len(frame) - 1, position + self.edge_margin)
            t = np.arange(start, stop + 1)
            rise = 0.5 - 0.5 * np.cos(np.pi * (t - start) / max(position - start, 1))
            fall = 0.5 + 0.5 * np.cos(np.pi * (t - position) / max(stop - position, 1))
            bump = np.where(t <= position, rise, fall)
            marked[start:stop + 1] += np.sign(carrier[position]) * (target - magnitude) * bump

        return fast + marked

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        audio = np.asarray(data, dtype=np.float64)
        frame_count = self._frame_count(len(audio), len(message))
        stego = audio.copy()
        dtype = np.asarray(data).dtype

        for index in range(frame_count):
            bit = int(message[index % len(message)])
            frame = audio[index * self.frame_length:(index + 1) * self.frame_length]

            marked = frame
            for _ in range(self.max_refinements):
                marked = self._mark(marked, bit)
                # Judge the frame as the decoder will receive it, after the
                # conversion back to the caller's sample type.
                votes = self._votes(marked.astype(dtype).astype(np.float64))
                if len(votes) and np.mean(votes == (1.0 if bit else -1.0)) >= self.AGREEMENT:
                    break

            stego[index * self.frame_length:(index + 1) * self.frame_length] = marked

        return stego.astype(dtype, copy=False)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        audio = np.asarray(data_with_watermark, dtype=np.float64)
        frame_count = self._frame_count(len(audio), watermark_length)

        scores = np.zeros(watermark_length)
        for index in range(frame_count):
            frame = audio[index * self.frame_length:(index + 1) * self.frame_length]
            votes = self._votes(frame)
            if len(votes):
                scores[index % watermark_length] += float(np.mean(votes))

        return [int(score > 0) for score in scores]

    def type(self) -> str:
        return "Empirical mode decomposition extrema quantisation (EMD)"
