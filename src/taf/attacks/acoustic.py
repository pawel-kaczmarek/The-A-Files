"""Echo, reverberation and acoustic-channel attacks.

Echo and reverberation are kept apart because they are different channels. A
single echo is one delayed copy,

    y[n] = x[n] + a * x[n - D]

which is a comb filter: it has a known cepstral signature and it is the direct
adversary of echo-hiding schemes, whose detector looks for exactly that
signature at a chosen delay. Reverberation is a dense decaying sequence of
reflections, which smears the signal in time without any single identifiable
delay, and damages block-synchronised methods rather than cepstral ones.

The acoustic channel composes the transformations that a signal played through
a loudspeaker and recaptured by a microphone actually undergoes. It is the
hardest realistic attack in this package, and the one most watermarking papers
call "over-the-air".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from taf.attacks.base import (
    Attack,
    AttackCategory,
    AttackError,
    clip_to_full_scale,
    per_channel,
)


def synthetic_impulse_response(
    rt60_seconds: float,
    sample_rate: int,
    seed: int | None = 0,
    direct_gain: float = 1.0,
    predelay_ms: float = 5.0,
) -> np.ndarray:
    """Exponentially decaying Gaussian noise as a synthetic room response.

    This is the standard textbook stand-in for a measured room impulse
    response: a direct path, a short pre-delay, then Gaussian noise whose
    envelope decays by 60 dB over RT60,

        h[n] = g[n] * 10^(-3 n / (RT60 * fs))

    It reproduces the decay rate and the diffuse character of a real room but
    not its modal structure or its frequency-dependent absorption, so it
    should be described as a synthetic response and not passed off as a
    measurement. It is fully determined by (rt60, sample_rate, seed), which a
    real recording never is.
    """
    if rt60_seconds <= 0:
        raise AttackError(f"rt60_seconds must be positive, got {rt60_seconds}")

    length = max(int(rt60_seconds * sample_rate), 2)
    rng = np.random.default_rng(seed)

    index = np.arange(length)
    # -60 dB over RT60 samples: amplitude factor 10^(-3n/(RT60*fs)).
    envelope = 10.0 ** (-3.0 * index / (rt60_seconds * sample_rate))
    response = rng.standard_normal(length) * envelope

    predelay = int(round(predelay_ms * sample_rate / 1000.0))
    response = np.concatenate([np.zeros(predelay), response])
    response[0] = direct_gain

    # Unit energy keeps the reverberated signal at the same level as the dry
    # one, so the attack does not smuggle in a gain change.
    energy = float(np.sqrt(np.sum(np.square(response))))
    return response / energy if energy > 0 else response


@dataclass(frozen=True)
class EchoAttack(Attack):
    """A single delayed copy: y[n] = x[n] + attenuation * x[n - D].

    Delays are given in milliseconds so they mean the same thing at any
    sample rate. Delays in the 5-100 ms range are the interesting ones: below
    roughly 10 ms the echo fuses with the direct sound and colours it, above
    50 ms it is heard as a separate repetition.

    An echo-hiding decoder searches the cepstrum for a peak at its own delay,
    so an added echo at a *different* delay inserts a competing peak. This
    attack is therefore targeted rather than generic, and should be reported
    as such.
    """

    delay_ms: float = 25.0
    attenuation: float = 0.25
    prevent_clipping: bool = True

    name = "echo"
    category = AttackCategory.ACOUSTIC

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        if self.delay_ms <= 0:
            raise AttackError(f"delay_ms must be positive, got {self.delay_ms}")
        if not 0.0 < self.attenuation <= 1.0:
            raise AttackError(f"attenuation must be in (0, 1], got {self.attenuation}")

        delay = int(round(self.delay_ms * sample_rate / 1000.0))
        if delay < 1:
            raise AttackError(
                f"delay_ms={self.delay_ms} is below one sample at {sample_rate} Hz"
            )
        if delay >= audio.shape[0]:
            raise AttackError(f"delay of {delay} samples exceeds the signal length")

        echoed = audio.copy()
        echoed[delay:] += self.attenuation * audio[:audio.shape[0] - delay]

        clipped = 0
        if self.prevent_clipping:
            echoed, clipped = clip_to_full_scale(echoed)

        return echoed, sample_rate, {
            "delay_ms": float(self.delay_ms),
            "delay_samples": delay,
            "attenuation": float(self.attenuation),
            "clipped_samples": clipped,
            "model": "single_reflection_comb",
        }


@dataclass(frozen=True)
class Reverberation(Attack):
    """Convolution with a synthetic room impulse response.

    Args:
        rt60_seconds: Decay time. 0.2 s is a small treated room, 0.5 s a
            living room or office, 1.0 s a hall - the range over which
            recorded audio is normally encountered.
        mix: Wet/dry ratio; 1.0 is fully reverberant.
        seed: Seed of the synthetic response, so a benchmark run repeats.
        trim_to_input: Convolution lengthens the signal by the response
            length. Trimming keeps the decoder's framing intact so the result
            measures reverberation rather than a length change.
    """

    rt60_seconds: float = 0.5
    mix: float = 1.0
    seed: int | None = 0
    trim_to_input: bool = True

    name = "reverb"
    category = AttackCategory.ACOUSTIC

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        if not 0.0 < self.mix <= 1.0:
            raise AttackError(f"mix must be in (0, 1], got {self.mix}")

        response = synthetic_impulse_response(self.rt60_seconds, sample_rate, self.seed)
        length = audio.shape[0]

        def convolve(channel: np.ndarray) -> np.ndarray:
            wet = fftconvolve(channel, response, mode="full")
            return wet[:length] if self.trim_to_input else wet

        wet_signal = per_channel(audio, convolve)

        if self.trim_to_input:
            mixed = (1.0 - self.mix) * audio + self.mix * wet_signal
        else:
            padded = np.zeros_like(wet_signal)
            padded[:length] = audio
            mixed = (1.0 - self.mix) * padded + self.mix * wet_signal

        return mixed, sample_rate, {
            "rt60_seconds": float(self.rt60_seconds),
            "mix": float(self.mix),
            "impulse_response": "synthetic_exponential_gaussian",
            "impulse_response_length": int(response.shape[0]),
            "seed": self.seed,
            "trimmed": bool(self.trim_to_input),
        }


@dataclass(frozen=True)
class AcousticChannel(Attack):
    """Loudspeaker, room and microphone in series.

    The chain is, in order:

    1. band limitation, since neither a loudspeaker nor a microphone
       reproduces the extremes of the band;
    2. convolution with a synthetic room response;
    3. additive background noise at a stated SNR;
    4. a gain change, since capture level is arbitrary;
    5. an optional clock mismatch between playback and capture devices.

    Each stage is configurable and each is also available as an attack in its
    own right; this composition exists because the stages interact, and a
    method can survive all five separately and fail the combination.

    This is a *simulation*. Real over-the-air capture also brings microphone
    nonlinearity, room modes, movement and ambient events that no synthetic
    chain reproduces. Results from it should be reported as a simulated
    acoustic path, never as a recording.
    """

    band_low_hz: float = 100.0
    band_high_hz: float = 7000.0
    rt60_seconds: float = 0.3
    snr_db: float = 25.0
    gain_db: float = -3.0
    clock_offset_ppm: float = 0.0
    seed: int | None = 0

    name = "acoustic_channel"
    category = AttackCategory.ACOUSTIC
    changes_length_or_rate = True

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        from taf.attacks.amplitude import GainChange
        from taf.attacks.filtering import BandPassFilter
        from taf.attacks.noise import AdditiveWhiteNoise
        from taf.attacks.resampling import SampleRateOffset

        stages: list[dict[str, Any]] = []
        signal = audio

        band = BandPassFilter(low_hz=self.band_low_hz, high_hz=self.band_high_hz)
        signal, _, meta = band._process(signal, sample_rate)
        stages.append({"stage": band.name, **meta})

        room = Reverberation(rt60_seconds=self.rt60_seconds, seed=self.seed)
        signal, _, meta = room._process(signal, sample_rate)
        stages.append({"stage": room.name, **meta})

        noise = AdditiveWhiteNoise(snr_db=self.snr_db, seed=self.seed)
        signal, _, meta = noise._process(signal, sample_rate)
        stages.append({"stage": noise.name, **meta})

        gain = GainChange(gain_db=self.gain_db, prevent_clipping=True)
        signal, _, meta = gain._process(signal, sample_rate)
        stages.append({"stage": gain.name, **meta})

        if self.clock_offset_ppm:
            drift = SampleRateOffset(offset_ppm=self.clock_offset_ppm)
            signal, _, meta = drift._process(signal, sample_rate)
            stages.append({"stage": drift.name, **meta})

        return signal, sample_rate, {
            "stages": stages,
            "simulated": True,
            "note": "synthetic playback-and-recapture chain, not a real recording",
        }


__all__ = ["AcousticChannel", "EchoAttack", "Reverberation", "synthetic_impulse_response"]
