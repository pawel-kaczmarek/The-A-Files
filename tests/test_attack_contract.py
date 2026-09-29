"""The contract every registered attack meets, checked on each one.

``test_attacks_dsp.py`` verifies what each attack does to a signal; this file
verifies what every attack promises regardless of what it does, so a newly
registered attack is covered without writing anything: it runs on real
speech with its defaults, leaves the caller's array alone, returns finite
audio with its parameters recorded, repeats exactly given its seed, and
resolves every severity level to explicit numbers that it accepts.
"""

from __future__ import annotations

import numpy as np
import pytest

from taf.attacks.base import AttackToolUnavailableError, Severity
from taf.attacks.registry import available_attacks, build
from taf.experiments.registry import list_attacks

SAMPLE_RATE = 16000

#: Attacks with no severity levels, and why. Every other attack needs a
#: mapping in ``taf.attacks.presets``.
WITHOUT_SEVERITY = {
    "zero_padding": "added after the benchmark suites were fixed; levels would change their composition",
}

_SPECS = {spec.name: spec for spec in list_attacks()}


def _apply(spec: str, audio: np.ndarray):
    try:
        return build(spec, SAMPLE_RATE).apply(audio, SAMPLE_RATE)
    except AttackToolUnavailableError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def cover(speech_cover: np.ndarray) -> np.ndarray:
    return np.asarray(speech_cover[: 2 * SAMPLE_RATE], dtype=np.float64)


@pytest.mark.parametrize("name", available_attacks())
def test_attack_runs_with_its_defaults_and_records_them(name: str, cover: np.ndarray):
    original = cover.copy()
    result = _apply(name, cover)

    assert np.array_equal(cover, original), "the attack wrote into the caller's array"
    assert np.all(np.isfinite(result.audio))
    assert result.audio.size > 0
    assert result.metadata["parameters"], "resolved parameters must reach the result row"
    if not _SPECS[name].changes_length_or_rate:
        assert result.audio.shape == cover.shape
        assert result.sample_rate == SAMPLE_RATE


@pytest.mark.parametrize("name", [name for name in available_attacks() if _SPECS[name].stochastic])
def test_stochastic_attack_repeats_given_its_seed(name: str, cover: np.ndarray):
    first = _apply(f"{name}:seed=11", cover)
    second = _apply(f"{name}:seed=11", cover)
    assert np.array_equal(first.audio, second.audio)
    assert first.metadata["parameters"]["seed"] == 11


@pytest.mark.parametrize("name", [name for name in available_attacks() if not _SPECS[name].has_severity])
def test_attack_without_severity_is_an_acknowledged_exception(name: str):
    assert name in WITHOUT_SEVERITY, f"{name} has no severity levels in taf.attacks.presets"


@pytest.mark.parametrize(
    "spec",
    [
        f"{name}@{severity.value}"
        for name in available_attacks()
        if _SPECS[name].has_severity
        for severity in Severity
    ],
)
def test_every_severity_resolves_to_parameters_the_attack_accepts(spec: str, cover: np.ndarray):
    result = _apply(spec, cover)
    assert np.all(np.isfinite(result.audio))
    assert result.metadata["parameters"]


def test_an_attack_can_declare_its_own_severity_levels(monkeypatch):
    """A plugin attack cannot edit the presets, so it declares its levels itself."""
    from dataclasses import dataclass

    from taf.attacks import presets, registry
    from taf.attacks.base import Attack, AttackCategory

    @dataclass(frozen=True)
    class ToyGain(Attack):
        gain_db: float = 0.0

        name = "toy_levels"
        category = AttackCategory.AMPLITUDE

        @classmethod
        def severity_levels(cls, severity, sample_rate):
            return {"gain_db": {"mild": -1.0, "moderate": -3.0, "strong": -6.0, "extreme": -12.0}[severity.value]}

        def _process(self, audio, sample_rate):
            return audio * 10 ** (self.gain_db / 20), sample_rate, {}

    monkeypatch.setattr(registry, "attack_classes", lambda: {**registry.ATTACK_CLASSES, "toy_levels": ToyGain})
    assert presets.severity_parameters("toy_levels", Severity.STRONG, SAMPLE_RATE) == {"gain_db": -6.0}
    assert registry.build("toy_levels@strong", SAMPLE_RATE).gain_db == -6.0


@pytest.mark.parametrize("rate", [8000, 11025, 16000, 22050, 32000, 44100])
def test_vorbis_ladders_encode_at_their_rate(rate: int):
    """libvorbis rejects bitrates outside a rate-dependent range instead of
    clamping them; every level and sweep value must be one it accepts."""
    from taf.attacks.presets import severity_parameters, sweep_presets

    bitrates = {severity_parameters("vorbis", level, rate)["bitrate_kbps"] for level in Severity}
    bitrates |= set(sweep_presets(rate)["vorbis"]["values"])
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * np.arange(rate // 4) / rate)
    for bitrate in sorted(bitrates):
        try:
            result = build(f"vorbis:bitrate_kbps={bitrate}").apply(tone, rate)
        except AttackToolUnavailableError as error:
            pytest.skip(str(error))
        assert np.all(np.isfinite(result.audio)), bitrate

