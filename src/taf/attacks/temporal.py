"""Time-domain and synchronisation attacks.

Most embedding schemes in this repository index the signal by absolute
position: frame *k* carries bit *k*. Anything that moves the sample grid -
a shift of a few samples, a cut, an inserted click, a 1% tempo change -
breaks that mapping, and the payload is lost even though every sample value
is still perfectly intact. These attacks are therefore the sharpest available
discriminator between position-based schemes and the few designs that survive
desynchronisation, such as the histogram method here.

Three operations that are often conflated are kept separate, because they are
different physics:

* **Time stretch** changes duration at constant pitch (phase vocoder).
* **Speed change** replays at a different rate, changing duration *and* pitch
  together, as a tape or a clock mismatch does.
* **Pitch shift** changes pitch at constant duration.
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
    from_columns,
    per_channel,
)
from taf.models.card import AttackCard, Text


@dataclass(frozen=True)
class TimeShift(Attack):
    """Displace the signal along the time axis.

    Args:
        shift_samples: Displacement; positive delays, negative advances.
            Takes precedence over ``shift_ms`` when both are given.
        shift_ms: Displacement in milliseconds, converted using the sample
            rate. Use this for values that should mean the same thing across
            datasets recorded at different rates.
        mode: ``"pad"`` shifts in zeros and keeps the length (what a delayed
            or trimmed transmission looks like), ``"circular"`` rotates the
            signal. Padding is the realistic default; circular shift is
            available because some published evaluations use it, and it keeps
            every sample present, which isolates pure desynchronisation.
    """

    shift_samples: int | None = None
    shift_ms: float | None = 10.0
    mode: str = "pad"

    name = "time_shift"
    category = AttackCategory.TEMPORAL
    card = AttackCard(
        title=Text("Time shift", "Przesunięcie w czasie"),
        summary=Text(
            en="Moves the recording in time to test decoder synchronisation.",
            pl="Przesuwa nagranie w czasie, sprawdzając synchronizację dekodera.",
        ),
        details=Text(
            en=(
                "Uses shift_samples or shift_ms to delay or advance audio. Pad mode fills exposed "
                "positions with zeros while keeping length; circular mode wraps samples around. A small "
                "shift can move every decoding frame away from its embedded position even when much of "
                "the waveform itself is unchanged."
            ),
            pl=(
                "Używa shift_samples lub shift_ms do opóźnienia albo przyspieszenia sygnału. Tryb pad "
                "uzupełnia odsłonięte pozycje zerami przy zachowaniu długości; circular przenosi próbki "
                "na drugi koniec. Małe przesunięcie może przestawić każdą ramkę odczytu, nawet jeśli "
                "większość przebiegu pozostaje niezmieniona."
            ),
        ),
    )
    changes_length_or_rate = True

    def _resolve_shift(self, sample_rate: int) -> int:
        if self.shift_samples is not None:
            return int(self.shift_samples)
        if self.shift_ms is None:
            raise AttackError("either shift_samples or shift_ms must be given")
        return int(round(self.shift_ms * sample_rate / 1000.0))

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        shift = self._resolve_shift(sample_rate)
        columns, was_mono = as_columns(audio)
        length = columns.shape[0]

        if abs(shift) >= length:
            raise AttackError(f"shift of {shift} samples exceeds the signal length ({length})")

        if self.mode == "circular":
            shifted = np.roll(columns, shift, axis=0)
        elif self.mode == "pad":
            shifted = np.zeros_like(columns)
            if shift > 0:
                shifted[shift:] = columns[:length - shift]
            elif shift < 0:
                shifted[:length + shift] = columns[-shift:]
            else:
                shifted = columns.copy()
        else:
            raise AttackError(f"unknown shift mode {self.mode!r}; expected 'pad' or 'circular'")

        return from_columns(shifted, was_mono), sample_rate, {
            "shift_samples": shift,
            "shift_ms": float(shift * 1000.0 / sample_rate),
            "shift_mode": self.mode,
            "samples_lost": 0 if self.mode == "circular" else abs(shift),
        }


@dataclass(frozen=True)
class Cropping(Attack):
    """Remove a contiguous piece of the signal.

    Args:
        fraction: Share of the signal removed, in (0, 1).
        position: ``"start"``, ``"end"``, ``"both"`` (split evenly) or
            ``"random"``. Fixed positions are preferred for a benchmark - the
            definition must not change between experiments - and ``"random"``
            is seeded so a run is still reproducible.
        seed: Seed for the random position.
    """

    fraction: float = 0.01
    position: str = "start"
    seed: int | None = 0

    name = "crop"
    category = AttackCategory.TEMPORAL
    card = AttackCard(
        title=Text("Cropping", "Przycięcie"),
        summary=Text(
            en="Removes part of the recording to test payload loss and framing sensitivity.",
            pl="Usuwa część nagrania, badając utratę danych i wrażliwość podziału na ramki.",
        ),
        details=Text(
            en=(
                "Deletes the configured fraction from the start, end, both ends or a seeded random "
                "location. The result is shorter, so some bits may disappear and length-derived frame "
                "boundaries may move. It tests partial-recording recovery; success depends on "
                "redundancy, synchronisation and where the method stored its message."
            ),
            pl=(
                "Wycina zadaną część z początku, końca, obu końców lub losowego miejsca ustalanego "
                "ziarnem. Wynik jest krótszy, więc część bitów może zniknąć, a granice ramek wyznaczane "
                "z długości mogą się przesunąć. Sprawdza odczyt z fragmentu nagrania; powodzenie zależy "
                "od nadmiarowości, synchronizacji i miejsca zapisu wiadomości."
            ),
        ),
    )
    changes_length_or_rate = True

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        if not 0.0 < self.fraction < 1.0:
            raise AttackError(f"fraction must be in (0, 1), got {self.fraction}")

        columns, was_mono = as_columns(audio)
        length = columns.shape[0]
        removed = int(round(length * self.fraction))
        if removed < 1:
            raise AttackError(
                f"fraction {self.fraction} removes no samples from a {length}-sample signal"
            )

        if self.position == "start":
            start = 0
        elif self.position == "end":
            start = length - removed
        elif self.position == "both":
            head = removed // 2
            kept = np.concatenate([columns[head:length - (removed - head)]], axis=0)
            return from_columns(kept, was_mono), sample_rate, {
                "removed_samples": removed,
                "removed_fraction": removed / length,
                "position": "both",
                "cut_start": 0,
            }
        elif self.position == "random":
            rng = np.random.default_rng(self.seed)
            start = int(rng.integers(0, length - removed + 1))
        else:
            raise AttackError(
                f"unknown crop position {self.position!r}; "
                "expected 'start', 'end', 'both' or 'random'"
            )

        kept = np.concatenate([columns[:start], columns[start + removed:]], axis=0)

        return from_columns(kept, was_mono), sample_rate, {
            "removed_samples": removed,
            "removed_fraction": removed / length,
            "position": self.position,
            "cut_start": int(start),
        }


@dataclass(frozen=True)
class ZeroPadding(Attack):
    """Prepend (and optionally append) silence.

    Everything after the padding moves, so a decoder that counts samples from
    the start of the file reads every frame at the wrong offset. Leading
    silence is also completely routine - it appears whenever a clip is cut
    from a longer recording or a container adds priming samples - which makes
    this one of the most realistic synchronisation attacks.
    """

    fraction: float = 0.1
    position: str = "start"

    name = "zero_padding"
    category = AttackCategory.TEMPORAL
    card = AttackCard(
        title=Text("Zero padding", "Dopełnienie ciszą"),
        summary=Text(
            en="Adds silence at the start, the end or both, without changing the retained samples.",
            pl="Dodaje ciszę na początku, na końcu lub z obu stron, nie zmieniając zachowanych próbek.",
        ),
        details=Text(
            en=(
                "Inserts zero-valued samples amounting to fraction of the signal length at position "
                "start, end or both. Leading silence moves every embedded position, so a decoder that "
                "counts samples from the start of the file reads each frame at the wrong offset; "
                "trailing silence changes only the total length, which matters to methods that derive "
                "their framing from it. Leading silence is routine when a clip is cut from a longer "
                "recording or a container adds priming samples."
            ),
            pl=(
                "Wstawia próbki zerowe o łącznej długości fraction długości sygnału w położeniu start, "
                "end lub both. Cisza na początku przesuwa wszystkie pozycje danych, więc dekoder "
                "liczący próbki od początku pliku czyta każdą ramkę z błędnym przesunięciem; cisza na "
                "końcu zmienia jedynie całkowitą długość, co ma znaczenie dla metod wyprowadzających z "
                "niej podział na ramki. Cisza na początku pojawia się rutynowo przy wycinaniu fragmentu "
                "nagrania lub gdy kontener dodaje próbki rozbiegowe."
            ),
        ),
    )
    changes_length_or_rate = True

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        if self.fraction <= 0:
            raise AttackError(f"fraction must be positive, got {self.fraction}")

        columns, was_mono = as_columns(audio)
        pad_length = int(round(columns.shape[0] * self.fraction))
        if pad_length < 1:
            raise AttackError(
                f"fraction {self.fraction} adds no samples to a {columns.shape[0]}-sample signal"
            )

        silence = np.zeros((pad_length, columns.shape[1]))
        if self.position == "start":
            padded = np.concatenate([silence, columns], axis=0)
        elif self.position == "end":
            padded = np.concatenate([columns, silence], axis=0)
        elif self.position == "both":
            padded = np.concatenate([silence, columns, silence], axis=0)
        else:
            raise AttackError(
                f"unknown padding position {self.position!r}; expected 'start', 'end' or 'both'"
            )

        return from_columns(padded, was_mono), sample_rate, {
            "padded_samples": int(pad_length),
            "position": self.position,
            "leading_offset": int(pad_length) if self.position in {"start", "both"} else 0,
        }


@dataclass(frozen=True)
class SampleInsertionDeletion(Attack):
    """Insert or delete short runs of samples at scattered positions.

    A jitter attack: the signal stays audibly identical while every block
    boundary after the first edit moves. Methods that re-derive their framing
    from the signal length, as most here do, lose alignment progressively
    along the clip rather than all at once.

    Deleted runs are cut out; inserted runs repeat the preceding sample, which
    is what a concealment algorithm does with a lost packet and avoids
    introducing a click that would be an attack of its own.
    """

    events: int = 10
    run_length: int = 8
    operation: str = "delete"
    seed: int | None = 0

    name = "sample_jitter"
    category = AttackCategory.TEMPORAL
    card = AttackCard(
        title=Text("Sample insertion and deletion", "Wstawianie i usuwanie próbek"),
        summary=Text(
            en="Inserts or deletes short sample runs at scattered positions.",
            pl="Wstawia lub usuwa krótkie serie próbek w rozproszonych miejscach.",
        ),
        details=Text(
            en=(
                "Deletion removes samples; insertion repeats preceding samples to model a simple "
                "concealment operation. Seeded positions make the attack reproducible. Each edit shifts "
                "subsequent boundaries, creating local and cumulative desynchronisation that a single "
                "global offset correction cannot necessarily repair."
            ),
            pl=(
                "Usuwanie wycina próbki, a wstawianie powtarza poprzednią próbkę, modelując proste "
                "maskowanie braków. Pozycje ustalane ziarnem zapewniają powtarzalność. Każda zmiana "
                "przesuwa kolejne granice, tworząc lokalną i narastającą desynchronizację, której nie "
                "musi naprawić jedno globalne przesunięcie."
            ),
        ),
    )
    changes_length_or_rate = True

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        if self.events < 1:
            raise AttackError(f"events must be positive, got {self.events}")
        if self.run_length < 1:
            raise AttackError(f"run_length must be positive, got {self.run_length}")
        if self.operation not in {"delete", "insert"}:
            raise AttackError(
                f"unknown operation {self.operation!r}; expected 'delete' or 'insert'"
            )

        columns, was_mono = as_columns(audio)
        length = columns.shape[0]
        total = self.events * self.run_length
        if total >= length:
            raise AttackError(
                f"{self.events} runs of {self.run_length} samples exceed the signal length"
            )

        rng = np.random.default_rng(self.seed)
        positions = np.sort(rng.choice(length - self.run_length, size=self.events, replace=False))

        pieces: list[np.ndarray] = []
        cursor = 0
        for position in positions:
            position = int(position)
            pieces.append(columns[cursor:position])
            if self.operation == "insert":
                anchor = columns[position:position + 1]
                pieces.append(np.repeat(anchor, self.run_length, axis=0))
                cursor = position
            else:
                cursor = position + self.run_length
        pieces.append(columns[cursor:])

        modified = np.concatenate(pieces, axis=0)

        return from_columns(modified, was_mono), sample_rate, {
            "events": int(self.events),
            "run_length": int(self.run_length),
            "operation": self.operation,
            "samples_changed": int(total),
            "length_delta": int(modified.shape[0] - length),
            "positions": [int(value) for value in positions],
        }


@dataclass(frozen=True)
class SampleDropout(Attack):
    """Zero out short runs of samples, keeping the length.

    Models packet loss or dropouts without concealment. Unlike cropping, the
    time axis is preserved, so this separates "the decoder lost alignment"
    from "the decoder lost data".
    """

    fraction: float = 0.01
    run_length: int = 20
    seed: int | None = 0

    name = "dropout"
    category = AttackCategory.TEMPORAL
    card = AttackCard(
        title=Text("Sample dropout", "Utrata próbek"),
        summary=Text(
            en="Replaces short audio regions with zeros, modelling missing packets or recording dropouts.",
            pl="Zastępuje krótkie obszary zerami, modelując utracone pakiety lub przerwy w nagraniu.",
        ),
        details=Text(
            en=(
                "Selects runs of samples and silences them while preserving total length. Parameters "
                "determine how much audio is lost and the run size; seed reproduces placement. Unlike "
                "cropping, positions after a dropout remain aligned, helping distinguish loss of "
                "payload content from loss of synchronisation."
            ),
            pl=(
                "Wybiera serie próbek i wycisza je, zachowując całkowitą długość. Parametry określają "
                "ilość traconego dźwięku i rozmiar serii, a seed odtwarza pozycje. W przeciwieństwie do "
                "przycięcia późniejsze próbki pozostają wyrównane, co pomaga odróżnić utratę treści "
                "danych od utraty synchronizacji."
            ),
        ),
    )

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        if not 0.0 < self.fraction < 1.0:
            raise AttackError(f"fraction must be in (0, 1), got {self.fraction}")
        if self.run_length < 1:
            raise AttackError(f"run_length must be positive, got {self.run_length}")

        columns, was_mono = as_columns(audio)
        length = columns.shape[0]
        runs = max(1, int(round(length * self.fraction / self.run_length)))

        rng = np.random.default_rng(self.seed)
        starts = rng.choice(max(1, length - self.run_length), size=runs, replace=False)

        modified = columns.copy()
        for start in starts:
            modified[int(start):int(start) + self.run_length] = 0.0

        zeroed = int(np.count_nonzero(np.all(modified == 0.0, axis=1)))

        return from_columns(modified, was_mono), sample_rate, {
            "runs": int(runs),
            "run_length": int(self.run_length),
            "requested_fraction": float(self.fraction),
            "zeroed_samples": zeroed,
        }


@dataclass(frozen=True)
class TimeStretch(Attack):
    """Change duration at constant pitch (phase vocoder).

    Note that a phase vocoder resynthesises the signal from STFT magnitudes
    with new phases; it does not merely reindex the samples. It is therefore a
    considerably harsher attack than a speed change of the same factor, and
    the two should never be reported as one.
    """

    rate: float = 1.01

    name = "time_stretch"
    category = AttackCategory.TEMPORAL
    card = AttackCard(
        title=Text("Time stretch", "Rozciągnięcie w czasie"),
        summary=Text(
            en="Changes duration while approximately preserving pitch.",
            pl="Zmienia czas trwania przy przybliżonym zachowaniu wysokości dźwięku.",
        ),
        details=Text(
            en=(
                "Uses a phase vocoder to resynthesise audio at a different tempo. This changes both the "
                "temporal grid and waveform phases, rather than merely moving existing samples. The "
                "rate controls the duration change. It can damage phase-based and frame-based payloads "
                "even when the tempo difference sounds small."
            ),
            pl=(
                "Używa wokodera fazowego do ponownej syntezy w innym tempie. Zmienia zarówno siatkę "
                "czasową, jak i fazy przebiegu, zamiast tylko przestawiać próbki. Współczynnik tempa "
                "ustala zmianę długości. Może uszkadzać dane fazowe i ramkowe nawet przy mało "
                "słyszalnej różnicy tempa."
            ),
        ),
    )
    changes_length_or_rate = True

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        import librosa

        if self.rate <= 0:
            raise AttackError(f"rate must be positive, got {self.rate}")

        stretched = per_channel(
            audio, lambda channel: librosa.effects.time_stretch(
                np.ascontiguousarray(channel, dtype=np.float32), rate=self.rate
            ).astype(np.float64)
        )

        return stretched, sample_rate, {
            "rate": float(self.rate),
            "algorithm": "librosa.effects.time_stretch (phase vocoder)",
            "pitch_preserved": True,
            "length_delta": int(stretched.shape[0] - audio.shape[0]),
        }


@dataclass(frozen=True)
class SpeedChange(Attack):
    """Replay faster or slower: duration and pitch change together.

    This is resampling the waveform onto a stretched grid while leaving the
    nominal sample rate alone - what a tape machine, a mismatched clock or a
    "1.05x playback" control does. It reindexes rather than resynthesises, so
    the waveform is preserved and amplitude-distribution methods survive it
    where they cannot survive a phase-vocoder stretch.
    """

    rate: float = 1.01

    name = "speed"
    category = AttackCategory.TEMPORAL
    card = AttackCard(
        title=Text("Speed change", "Zmiana prędkości"),
        summary=Text(
            en="Changes playback speed, affecting both duration and pitch.",
            pl="Zmienia prędkość odtwarzania, wpływając jednocześnie na długość i wysokość dźwięku.",
        ),
        details=Text(
            en=(
                "Resamples the waveform to model faster or slower playback while keeping the nominal "
                "output rate. Faster playback shortens the clip and raises pitch; slower playback does "
                "the reverse. It probes timing and frequency dependence together, unlike time "
                "stretching, which aims to preserve pitch."
            ),
            pl=(
                "Próbkuje przebieg ponownie, modelując szybsze lub wolniejsze odtwarzanie przy "
                "zachowaniu nominalnej częstotliwości wyjściowej. Szybsze odtwarzanie skraca nagranie i "
                "podnosi wysokość, a wolniejsze działa odwrotnie. Bada jednocześnie zależność czasową i "
                "częstotliwościową, inaczej niż zmiana tempa z zachowaniem wysokości."
            ),
        ),
    )
    changes_length_or_rate = True

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        from taf.attacks.resampling import resample_to

        if self.rate <= 0:
            raise AttackError(f"rate must be positive, got {self.rate}")

        # Playing back `rate` times faster is the same as resampling to
        # fs/rate and then declaring the result to be at fs.
        target = int(round(sample_rate / self.rate))
        if target <= 0:
            raise AttackError(f"rate {self.rate} is too large for a {sample_rate} Hz signal")

        changed = resample_to(audio, sample_rate, target)

        return changed, sample_rate, {
            "rate": float(self.rate),
            "effective_resample_hz": target,
            "algorithm": "scipy.signal.resample_poly",
            "pitch_preserved": False,
            "length_delta": int(changed.shape[0] - audio.shape[0]),
        }


@dataclass(frozen=True)
class PitchShift(Attack):
    """Change pitch at constant duration.

    Parameterised in semitones, including fractional ones: a quarter of a
    semitone is inaudible on speech yet completely reorganises the spectrum
    for any method reading fixed frequency bins.
    """

    semitones: float = 0.5

    name = "pitch_shift"
    category = AttackCategory.TEMPORAL
    card = AttackCard(
        title=Text("Pitch shift", "Zmiana wysokości dźwięku"),
        summary=Text(
            en="Raises or lowers pitch while keeping approximately the same duration.",
            pl="Podnosi lub obniża wysokość dźwięku przy zachowaniu przybliżonej długości.",
        ),
        details=Text(
            en=(
                "Applies a pitch-shifting operation parameterised in semitones. The resynthesis changes "
                "frequency content and phase relationships even though the overall duration is "
                "maintained. It tests whether embedding relies on particular spectral positions or "
                "waveform detail, rather than simply on clip length."
            ),
            pl=(
                "Stosuje zmianę wysokości określoną w półtonach. Ponowna synteza zmienia zawartość "
                "częstotliwościową i relacje faz mimo zachowania czasu trwania. Sprawdza, czy osadzanie "
                "zależy od konkretnych pozycji widma lub szczegółów przebiegu, a nie tylko od długości "
                "nagrania."
            ),
        ),
    )

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        import librosa

        shifted = per_channel(
            audio, lambda channel: librosa.effects.pitch_shift(
                np.ascontiguousarray(channel, dtype=np.float32),
                sr=sample_rate,
                n_steps=float(self.semitones),
            ).astype(np.float64)
        )

        return shifted, sample_rate, {
            "semitones": float(self.semitones),
            "frequency_ratio": float(2.0 ** (self.semitones / 12.0)),
            "algorithm": "librosa.effects.pitch_shift",
            "duration_preserved": True,
        }


__all__ = [
    "Cropping",
    "PitchShift",
    "SampleDropout",
    "SampleInsertionDeletion",
    "SpeedChange",
    "TimeShift",
    "TimeStretch",
    "ZeroPadding",
]
