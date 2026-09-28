"""Empirical mode decomposition (Huang et al., 1998) with a fixed sifting schedule.

Sifting stops after a fixed number of iterations rather than on a data
dependent criterion, so the decomposition is a deterministic function of the
samples: the embedder and a fresh decoder compute exactly the same modes.
"""
from typing import List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


def extrema(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of the interior local maxima and minima."""
    slope = np.sign(np.diff(signal))
    maxima = np.where((slope[:-1] > 0) & (slope[1:] <= 0))[0] + 1
    minima = np.where((slope[:-1] < 0) & (slope[1:] >= 0))[0] + 1
    return maxima, minima


def _envelope(signal: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Cubic-spline envelope through ``knots``, mirrored at both ends.

    Mirroring the outermost extremum about the frame edge keeps the spline
    from swinging freely past the last knot, the usual EMD end effect.
    """
    last = len(signal) - 1
    positions = np.concatenate(([-knots[0]], knots, [2 * last - knots[-1]]))
    values = np.concatenate(([signal[knots[0]]], signal[knots], [signal[knots[-1]]]))
    return CubicSpline(positions, values)(np.arange(len(signal)))


def emd(signal: np.ndarray, max_imfs: int, sift_iterations: int = 4) -> Tuple[List[np.ndarray], np.ndarray]:
    """Split ``signal`` into at most ``max_imfs`` intrinsic mode functions.

    Returns:
        The IMFs, highest frequency first, and the residue; their sum is the
        input signal.
    """
    residue = np.asarray(signal, dtype=np.float64).copy()
    imfs: List[np.ndarray] = []

    for _ in range(max_imfs):
        maxima, minima = extrema(residue)
        if len(maxima) < 2 or len(minima) < 2:
            break

        mode = residue.copy()
        for _ in range(sift_iterations):
            maxima, minima = extrema(mode)
            if len(maxima) < 2 or len(minima) < 2:
                break
            mode -= (_envelope(mode, maxima) + _envelope(mode, minima)) / 2.0

        imfs.append(mode)
        residue = residue - mode

    return imfs, residue
