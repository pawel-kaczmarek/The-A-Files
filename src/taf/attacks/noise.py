"""Additive noise attacks, parameterised by target SNR.

Noise level is specified as a signal-to-noise ratio rather than an absolute
amplitude. An absolute standard deviation makes the severity of the attack
depend on how loud the recording happens to be, so the same setting is a
scratch on one file and a destroyed signal on another, and results cannot be
compared across a dataset. With a target SNR the noise power is derived from
the measured signal power:

    P_signal = mean(x^2)
    P_noise  = P_signal / 10^(SNR_dB / 10)

The three noise processes are kept apart because they model different things
and different embedding domains resist them differently: white noise is the
generic channel, pink noise concentrates its energy where speech and music
live, and impulse noise is sparse and heavy-tailed, which is what clicks,
dropouts and bit errors look like.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from taf.attacks.base import (
    Attack,
    AttackCategory,
    AttackError,
    as_columns,
    clip_to_full_scale,
    from_columns,
    signal_power,
)
from taf.models.card import AttackCard, Text


def _noise_scale(audio: np.ndarray, snr_db: float) -> float:
    """Amplitude scale giving unit-variance noise the requested SNR."""
    power = signal_power(audio)
    if power == 0.0:
        raise AttackError("cannot set an SNR against a silent signal")
    return float(np.sqrt(power / (10.0 ** (snr_db / 10.0))))


def _pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance pink (1/f) noise by spectral shaping of white noise.

    Shaping in the frequency domain gives an exact 1/f power law, unlike the
    usual cascaded-filter approximations, and is deterministic given the seed.
    """
    spectrum = np.fft.rfft(rng.standard_normal(length))
    frequencies = np.arange(len(spectrum), dtype=np.float64)
    # Leave the DC bin alone; 1/sqrt(f) in amplitude is 1/f in power.
    frequencies[0] = 1.0
    shaped = spectrum / np.sqrt(frequencies)
    noise = np.fft.irfft(shaped, n=length)

    deviation = float(np.std(noise))
    if deviation == 0.0:
        return noise
    return noise / deviation


@dataclass(frozen=True)
class AdditiveWhiteNoise(Attack):
    """Additive white Gaussian noise at a controlled SNR.

    Models the generic transmission or recording channel, and is the standard
    stress test for spread-spectrum and quantisation-based embedding, whose
    detection statistics are sums over many samples.
    """

    snr_db: float = 20.0
    seed: int | None = 0
    prevent_clipping: bool = False

    name = "awgn"
    category = AttackCategory.NOISE
    card = AttackCard(
        title=Text("White Gaussian noise", "Biały szum gaussowski"),
        summary=Text(
            en="Adds white Gaussian noise to model a noisy recording or transmission channel.",
            pl="Dodaje biały szum gaussowski, symulując zaszumione nagranie lub kanał transmisyjny.",
        ),
        details=Text(
            en=(
                "Measures signal power and scales random noise to the requested snr_db. Lower SNR means "
                "stronger noise, while seed reproduces the random sequence. Noise perturbs sample "
                "values and transform coefficients throughout the recording; the resulting bit errors "
                "reveal the detector’s tolerance to distributed disturbance."
            ),
            pl=(
                "Mierzy moc sygnału i skaluje losowy szum do zadanego snr_db. Niższy SNR oznacza "
                "silniejszy szum, a seed pozwala odtworzyć sekwencję. Szum zmienia wartości próbek i "
                "współczynniki transformacji w całym nagraniu; błędy bitowe pokazują tolerancję "
                "dekodera na rozproszone zakłócenie."
            ),
        ),
    )

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        scale = _noise_scale(audio, self.snr_db)
        noise = rng.standard_normal(audio.shape) * scale
        noisy = audio + noise

        clipped = 0
        if self.prevent_clipping:
            noisy, clipped = clip_to_full_scale(noisy)

        return noisy, sample_rate, {
            "requested_snr_db": self.snr_db,
            "noise_rms": float(np.sqrt(signal_power(noise))),
            "clipped_samples": clipped,
            "noise_process": "gaussian_white",
        }


@dataclass(frozen=True)
class AdditivePinkNoise(Attack):
    """Additive pink (1/f) noise at a controlled SNR.

    Pink noise puts most of its power at low frequencies, where speech and
    music energy also sits, so at equal SNR it masks low-band embedding far
    more than white noise does.
    """

    snr_db: float = 20.0
    seed: int | None = 0
    prevent_clipping: bool = False

    name = "pink_noise"
    category = AttackCategory.NOISE
    card = AttackCard(
        title=Text("Pink noise", "Szum różowy"),
        summary=Text(
            en="Adds pink noise, concentrating more disturbance at low frequencies than white noise.",
            pl="Dodaje szum różowy, skupiający więcej zakłóceń w niskich częstotliwościach niż szum biały.",
        ),
        details=Text(
            en=(
                "Generates noise with an approximately 1/f power spectrum and scales it to the "
                "requested SNR. snr_db controls total power and seed controls reproducibility. At equal "
                "SNR, its frequency distribution differs from AWGN, so comparing them helps identify "
                "whether the payload depends on low-frequency content."
            ),
            pl=(
                "Generuje szum o widmie mocy zbliżonym do 1/f i skaluje go do zadanego SNR. snr_db "
                "steruje mocą, a seed powtarzalnością. Przy równym SNR rozkład częstotliwości różni się "
                "od AWGN, więc porównanie pokazuje zależność danych od niskich częstotliwości."
            ),
        ),
    )

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        scale = _noise_scale(audio, self.snr_db)

        columns, was_mono = as_columns(audio)
        noise = np.stack(
            [_pink_noise(columns.shape[0], rng) for _ in range(columns.shape[1])], axis=1
        ) * scale
        noisy = from_columns(columns + noise, was_mono)

        clipped = 0
        if self.prevent_clipping:
            noisy, clipped = clip_to_full_scale(noisy)

        return noisy, sample_rate, {
            "requested_snr_db": self.snr_db,
            "noise_rms": float(np.sqrt(signal_power(noise))),
            "clipped_samples": clipped,
            "noise_process": "pink_1_over_f",
        }


@dataclass(frozen=True)
class ImpulseNoise(Attack):
    """Sparse high-amplitude impulses at a controlled SNR.

    Models clicks, dropouts and bit errors: a small fraction of samples is hit
    hard while the rest is untouched. Sample-domain embedding suffers in the
    positions that are hit; transform-domain embedding spreads the damage
    thinly across coefficients and mostly shrugs it off.

    ``density`` is the fraction of samples affected; the SNR then fixes how
    large each impulse is, so severity stays comparable across files.
    """

    snr_db: float = 20.0
    density: float = 0.001
    seed: int | None = 0
    prevent_clipping: bool = False

    name = "impulse_noise"
    category = AttackCategory.NOISE
    card = AttackCard(
        title=Text("Impulse noise", "Szum impulsowy"),
        summary=Text(
            en="Injects sparse strong impulses, resembling clicks or isolated transmission errors.",
            pl="Dodaje rzadkie silne impulsy przypominające trzaski lub pojedyncze błędy transmisji.",
        ),
        details=Text(
            en=(
                "Randomly selects a fraction of samples using density and sets impulse power from "
                "snr_db. At fixed total noise power, fewer affected samples mean stronger individual "
                "impulses. seed reproduces their positions. This distinguishes vulnerability to local "
                "damage from sensitivity to continuous background noise."
            ),
            pl=(
                "Losowo wybiera część próbek według density i ustala moc impulsów przez snr_db. Przy "
                "stałej mocy zakłócenia mniej trafionych próbek oznacza silniejsze pojedyncze impulsy. "
                "seed odtwarza ich pozycje. Atak pozwala odróżnić podatność na lokalne uszkodzenia od "
                "wrażliwości na ciągły szum."
            ),
        ),
    )

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        if not 0.0 < self.density <= 1.0:
            raise AttackError(f"density must be in (0, 1], got {self.density}")

        rng = np.random.default_rng(self.seed)
        columns, was_mono = as_columns(audio)

        mask = rng.random(columns.shape) < self.density
        hit_count = int(np.count_nonzero(mask))
        if hit_count == 0:
            return audio, sample_rate, {
                "requested_snr_db": self.snr_db,
                "impulses": 0,
                "noise_process": "impulsive",
                "clipped_samples": 0,
            }

        # Total impulse energy is fixed by the SNR and spread over the samples
        # that were hit, so a sparser attack means taller impulses.
        target_noise_power = signal_power(columns) / (10.0 ** (self.snr_db / 10.0))
        amplitude = float(np.sqrt(target_noise_power * columns.size / hit_count))

        noise = np.zeros_like(columns)
        signs = rng.choice((-1.0, 1.0), size=hit_count)
        noise[mask] = signs * amplitude

        noisy = from_columns(columns + noise, was_mono)

        clipped = 0
        if self.prevent_clipping:
            noisy, clipped = clip_to_full_scale(noisy)

        return noisy, sample_rate, {
            "requested_snr_db": self.snr_db,
            "impulses": hit_count,
            "impulse_amplitude": amplitude,
            "clipped_samples": clipped,
            "noise_process": "impulsive",
        }


__all__ = ["AdditiveWhiteNoise", "AdditivePinkNoise", "ImpulseNoise"]
