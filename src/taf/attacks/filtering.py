"""Linear filtering attacks.

Filtering removes a band outright, so it is the sharpest test of *where* a
method puts its payload. Anything embedded above 4 kHz is gone after a 4 kHz
low-pass no matter how cleverly it was encoded, while a method carrying its
payload in the low band is untouched.

Two design points are stated explicitly rather than left implicit, because
they change the outcome for synchronisation-sensitive methods:

* **Filter family and order.** Butterworth in second-order-section form:
  maximally flat in the passband, numerically stable at high order, and
  specified by two numbers. Order 6 (36 dB/octave) is the default - steep
  enough to be a real band limit, gentle enough to stay a plausible piece of
  audio processing.
* **Phase.** ``zero_phase=True`` (the default) applies the filter forwards and
  backwards, which doubles the attenuation and leaves no group delay. Causal
  filtering shifts the signal in a frequency-dependent way, which desynchronises
  position-based decoders - a real effect, but one that is easy to mistake for
  the band limit itself. Both are available and the choice is recorded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, filtfilt, lfilter

from taf.attacks.base import (
    Attack,
    AttackCategory,
    AttackError,
    per_channel,
    validate_cutoff,
)
from taf.models.card import AttackCard, Text


def _apply_sos(audio: np.ndarray, sos: np.ndarray, zero_phase: bool) -> np.ndarray:
    if zero_phase:
        return per_channel(audio, lambda channel: sosfiltfilt(sos, channel))
    return per_channel(audio, lambda channel: sosfilt(sos, channel))


@dataclass(frozen=True)
class LowPassFilter(Attack):
    """Remove everything above a cutoff frequency."""

    cutoff_hz: float = 4000.0
    order: int = 6
    zero_phase: bool = True

    name = "low_pass"
    category = AttackCategory.FILTERING
    card = AttackCard(
        title=Text("Low-pass filter", "Filtr dolnoprzepustowy"),
        summary=Text(
            en="Attenuates high frequencies to test whether the watermark depends on the upper band.",
            pl="Tłumi wysokie częstotliwości, sprawdzając zależność znaku od górnego pasma.",
        ),
        details=Text(
            en=(
                "Uses a Butterworth low-pass filter specified by cutoff and order. The default "
                "forward-backward mode avoids group delay but applies filtering twice; causal mode also "
                "changes phase. Lower cutoff removes more bandwidth. Lost upper-band payload components "
                "cannot be recovered merely by restoring the original sample rate."
            ),
            pl=(
                "Stosuje filtr dolnoprzepustowy Butterwortha określony częstotliwością odcięcia i "
                "rzędem. Domyślny przebieg w przód i wstecz usuwa opóźnienie grupowe, ale filtruje "
                "dwukrotnie; wariant przyczynowy zmienia także fazę. Niższe odcięcie usuwa więcej "
                "pasma. Przywrócenie częstotliwości próbkowania nie odtwarza utraconych składowych "
                "danych."
            ),
        ),
    )

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        validate_cutoff(self.cutoff_hz, sample_rate)
        sos = butter(self.order, self.cutoff_hz, fs=sample_rate, btype="low", output="sos")
        filtered = _apply_sos(audio, sos, self.zero_phase)

        return filtered, sample_rate, {
            "cutoff_hz": float(self.cutoff_hz),
            "order": int(self.order),
            "family": "butterworth",
            "zero_phase": bool(self.zero_phase),
            "effective_order": int(self.order * (2 if self.zero_phase else 1)),
        }


@dataclass(frozen=True)
class HighPassFilter(Attack):
    """Remove everything below a cutoff frequency.

    Relevant to methods that hide in the low band - amplitude-relation and
    histogram schemes in this repository do - and a routine step in any
    recording chain, where it removes rumble and DC.
    """

    cutoff_hz: float = 300.0
    order: int = 6
    zero_phase: bool = True

    name = "high_pass"
    category = AttackCategory.FILTERING
    card = AttackCard(
        title=Text("High-pass filter", "Filtr górnoprzepustowy"),
        summary=Text(
            en="Attenuates low frequencies, as in rumble removal or an AC-coupled recording path.",
            pl="Tłumi niskie częstotliwości, podobnie do usuwania dudnienia lub toru bez składowej stałej.",
        ),
        details=Text(
            en=(
                "Applies a Butterworth high-pass filter with a chosen cutoff and order. Raising the "
                "cutoff removes more low-band content and can damage amplitude-relation or "
                "low-frequency transform embedding. zero_phase controls forward-backward versus causal "
                "filtering, separating band attenuation from phase-delay effects."
            ),
            pl=(
                "Stosuje filtr górnoprzepustowy Butterwortha o wybranym odcięciu i rzędzie. "
                "Podniesienie odcięcia usuwa więcej niskiego pasma i może niszczyć osadzanie oparte na "
                "relacjach amplitud lub niskich częstotliwościach. zero_phase wybiera filtrację w przód "
                "i wstecz albo przyczynową, rozdzielając tłumienie pasma od efektów opóźnienia "
                "fazowego."
            ),
        ),
    )

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        validate_cutoff(self.cutoff_hz, sample_rate)
        sos = butter(self.order, self.cutoff_hz, fs=sample_rate, btype="high", output="sos")
        filtered = _apply_sos(audio, sos, self.zero_phase)

        return filtered, sample_rate, {
            "cutoff_hz": float(self.cutoff_hz),
            "order": int(self.order),
            "family": "butterworth",
            "zero_phase": bool(self.zero_phase),
            "effective_order": int(self.order * (2 if self.zero_phase else 1)),
        }


@dataclass(frozen=True)
class BandPassFilter(Attack):
    """Keep only a band, discarding everything outside it.

    Models a band-limited transmission channel - telephony being the obvious
    case at roughly 300 Hz to 3.4 kHz.
    """

    low_hz: float = 300.0
    high_hz: float = 3400.0
    order: int = 6
    zero_phase: bool = True

    name = "band_pass"
    category = AttackCategory.FILTERING
    card = AttackCard(
        title=Text("Band-pass filter", "Filtr pasmowoprzepustowy"),
        summary=Text(
            en="Keeps only a selected frequency band, modelling a bandwidth-limited channel.",
            pl=(
                "Przepuszcza tylko wybrane pasmo częstotliwości, modelując kanał o ograniczonej "
                "szerokości pasma."
            ),
        ),
        details=Text(
            en=(
                "Combines lower and upper cutoffs in a Butterworth band-pass filter. Narrowing the "
                "passband removes more carrier information, as in telephone-like transmission. Filter "
                "order controls transition steepness and zero_phase controls phase handling. Payload "
                "stored outside the retained band is especially exposed."
            ),
            pl=(
                "Łączy dolne i górne odcięcie w filtrze pasmowoprzepustowym Butterwortha. Zwężenie "
                "pasma usuwa więcej informacji nośnika, jak w transmisji telefonicznej. Rząd określa "
                "stromość zboczy, a zero_phase sposób obsługi fazy. Szczególnie narażone są dane "
                "zapisane poza zachowanym pasmem."
            ),
        ),
    )

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        validate_cutoff(self.low_hz, sample_rate, "low_hz")
        validate_cutoff(self.high_hz, sample_rate, "high_hz")
        if self.low_hz >= self.high_hz:
            raise AttackError(f"low_hz ({self.low_hz}) must be below high_hz ({self.high_hz})")

        sos = butter(
            self.order, (self.low_hz, self.high_hz), fs=sample_rate, btype="band", output="sos"
        )
        filtered = _apply_sos(audio, sos, self.zero_phase)

        return filtered, sample_rate, {
            "low_hz": float(self.low_hz),
            "high_hz": float(self.high_hz),
            "order": int(self.order),
            "family": "butterworth",
            "zero_phase": bool(self.zero_phase),
        }


@dataclass(frozen=True)
class NotchFilter(Attack):
    """Attenuate a narrow band around a centre frequency.

    This replaces the old ``frequency_filter``, which zeroed FFT bins whose
    frequency compared exactly equal to a constant - a test that essentially
    never fires on a floating-point frequency grid, so the attack was close to
    a no-op. A second-order IIR notch is the standard way to remove a tone
    (mains hum, a pilot, a carrier), and it has a defined bandwidth.

    ``depth_db`` allows partial attenuation: at ``None`` the notch is a full
    null, otherwise the removed component is added back scaled so the band is
    reduced by exactly that many decibels.
    """

    center_hz: float = 1000.0
    quality: float = 30.0
    depth_db: float | None = None

    name = "notch"
    category = AttackCategory.FILTERING
    card = AttackCard(
        title=Text("Notch filter", "Filtr zaporowy"),
        summary=Text(
            en="Suppresses a narrow frequency region, such as hum or a tonal interferer.",
            pl="Wycisza wąski zakres częstotliwości, np. przydźwięk lub zakłócenie tonalne.",
        ),
        details=Text(
            en=(
                "Uses a second-order IIR notch around a centre frequency; its quality factor determines "
                "bandwidth. Optional depth_db allows partial attenuation instead of a full notch. It "
                "probes whether a mark relies on a narrow carrier region rather than broadly "
                "distributed energy."
            ),
            pl=(
                "Używa filtra zaporowego IIR drugiego rzędu wokół częstotliwości środkowej; dobroć "
                "określa szerokość pasma. Opcjonalny depth_db pozwala na częściowe tłumienie zamiast "
                "pełnej zapory. Sprawdza, czy znak zależy od wąskiego obszaru widma, czy od szerzej "
                "rozłożonej energii."
            ),
        ),
    )

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        validate_cutoff(self.center_hz, sample_rate, "center_hz")
        if self.quality <= 0:
            raise AttackError(f"quality must be positive, got {self.quality}")

        b, a = iirnotch(self.center_hz, self.quality, fs=sample_rate)
        notched = per_channel(audio, lambda channel: filtfilt(b, a, channel))

        metadata: dict[str, Any] = {
            "center_hz": float(self.center_hz),
            "quality": float(self.quality),
            "bandwidth_hz": float(self.center_hz / self.quality),
            "zero_phase": True,
            "family": "iir_notch",
        }

        if self.depth_db is not None:
            if self.depth_db <= 0:
                raise AttackError(f"depth_db must be positive, got {self.depth_db}")
            # notched = x - band; scaling the removed band back in gives a
            # partial notch of the requested depth.
            retained = 10.0 ** (-self.depth_db / 20.0)
            band = audio - notched
            notched = notched + retained * band
            metadata["depth_db"] = float(self.depth_db)
            metadata["retained_band_gain"] = float(retained)

        return notched, sample_rate, metadata


@dataclass(frozen=True)
class MovingAverageSmoothing(Attack):
    """Unweighted moving-average (boxcar FIR) smoothing.

    A crude low-pass with a sinc-shaped response, kept as its own attack
    because it is what naive "denoising" code actually does, and because its
    nulls at multiples of fs/window_length interact with any embedding that
    places energy near those frequencies.
    """

    window_length: int = 5
    zero_phase: bool = True

    name = "smoothing"
    category = AttackCategory.FILTERING
    card = AttackCard(
        title=Text("Moving-average smoothing", "Wygładzanie średnią ruchomą"),
        summary=Text(
            en="Smooths samples by replacing them with local moving averages.",
            pl="Wygładza próbki przez zastępowanie ich lokalnymi średnimi ruchomymi.",
        ),
        details=Text(
            en=(
                "Convolves the waveform with a uniform boxcar window. A longer window suppresses fast "
                "variations more strongly and produces spectral nulls related to the window length. "
                "This models simple smoothing or naive denoising and can erase fine sample-level "
                "payload changes."
            ),
            pl=(
                "Wykonuje splot przebiegu z równomiernym oknem prostokątnym. Dłuższe okno mocniej tłumi "
                "szybkie zmiany i wprowadza zera widma zależne od długości okna. Modeluje proste "
                "wygładzanie lub naiwne odszumianie i może usuwać drobne zmiany próbek niosące "
                "wiadomość."
            ),
        ),
    )

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        if self.window_length < 2:
            raise AttackError(f"window_length must be at least 2, got {self.window_length}")

        taps = np.ones(int(self.window_length)) / float(self.window_length)
        if self.zero_phase:
            smoothed = per_channel(audio, lambda channel: filtfilt(taps, [1.0], channel))
        else:
            smoothed = per_channel(audio, lambda channel: lfilter(taps, [1.0], channel))

        return smoothed, sample_rate, {
            "window_length": int(self.window_length),
            "zero_phase": bool(self.zero_phase),
            "first_null_hz": float(sample_rate / self.window_length),
            "family": "boxcar_fir",
        }


__all__ = [
    "LowPassFilter",
    "HighPassFilter",
    "BandPassFilter",
    "NotchFilter",
    "MovingAverageSmoothing",
]
