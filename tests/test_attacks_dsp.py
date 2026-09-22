"""Signal-processing correctness tests for the attack package.

These assert what each attack does to the signal - the achieved SNR, the
stopband attenuation, the number of quantisation levels, the exact sample
displacement - rather than merely that a function returned an array. An attack
that silently does nothing (as the old frequency_filter did) passes a
type check and fails these.
"""
from __future__ import annotations

import numpy as np
import pytest

from taf.attacks import build
from taf.attacks.base import AttackError, measured_snr_db, rms
from taf.attacks.codec import ffmpeg_available
from taf.attacks.presets import Severity, benchmark_suite, severity_parameters

SAMPLE_RATE = 16000
needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not installed")


@pytest.fixture(scope="module")
def tone() -> np.ndarray:
    t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    return (0.4 * np.sin(2 * np.pi * 1000 * t) + 0.2 * np.sin(2 * np.pi * 3000 * t))


@pytest.fixture(scope="module")
def noise_signal() -> np.ndarray:
    return np.random.default_rng(1234).normal(0, 0.1, SAMPLE_RATE * 2)


def band_energy(signal: np.ndarray, low: float, high: float, sample_rate: int = SAMPLE_RATE) -> float:
    """Energy in a frequency band, measured through a Hann window.

    The window matters: an unwindowed DFT of a broadband signal leaks energy
    from the passband into the stopband through its sidelobes, which puts a
    floor of roughly -30 dB on any attenuation measurement and would make a
    perfectly good filter look like a failure.
    """
    windowed = np.asarray(signal, dtype=np.float64) * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    return float(spectrum[(frequencies >= low) & (frequencies < high)].sum())


# ------------------------------------------------------------------- noise


@pytest.mark.parametrize("target_snr", [40.0, 30.0, 20.0, 10.0, 5.0])
def test_awgn_achieves_the_requested_snr(noise_signal: np.ndarray, target_snr: float) -> None:
    result = build(f"awgn:snr_db={target_snr}").apply(noise_signal, SAMPLE_RATE)
    achieved = measured_snr_db(noise_signal, result.audio)
    assert achieved == pytest.approx(target_snr, abs=0.5)
    assert result.metadata["requested_snr_db"] == target_snr


def test_awgn_is_reproducible_and_seed_dependent(noise_signal: np.ndarray) -> None:
    first = build("awgn:snr_db=20,seed=7").apply(noise_signal, SAMPLE_RATE).audio
    again = build("awgn:snr_db=20,seed=7").apply(noise_signal, SAMPLE_RATE).audio
    other = build("awgn:snr_db=20,seed=8").apply(noise_signal, SAMPLE_RATE).audio

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_pink_noise_holds_equal_energy_per_octave(noise_signal: np.ndarray) -> None:
    """The defining property of 1/f noise, and what separates it from white."""
    attacked = build("pink_noise:snr_db=10").apply(noise_signal, SAMPLE_RATE).audio
    # Isolate the added noise; the carrier's own spectrum is not 1/f.
    added = attacked - noise_signal

    octaves = [band_energy(added, low, 2 * low) for low in (125, 250, 500, 1000, 2000)]
    assert max(octaves) / min(octaves) < 2.0

    white = build("awgn:snr_db=10").apply(noise_signal, SAMPLE_RATE).audio - noise_signal
    white_octaves = [band_energy(white, low, 2 * low) for low in (125, 250, 500, 1000, 2000)]
    # White noise doubles its band energy every octave; pink does not.
    assert white_octaves[-1] / white_octaves[0] > 8.0


def test_impulse_noise_is_sparse(noise_signal: np.ndarray) -> None:
    result = build("impulse_noise:snr_db=20,density=0.001").apply(noise_signal, SAMPLE_RATE)
    changed = int(np.count_nonzero(result.audio != noise_signal))

    assert changed == result.metadata["impulses"]
    assert 0 < changed < 0.01 * len(noise_signal)


def test_noise_rejects_a_silent_signal() -> None:
    with pytest.raises(AttackError):
        build("awgn:snr_db=20").apply(np.zeros(1000), SAMPLE_RATE)


# --------------------------------------------------------------- amplitude


@pytest.mark.parametrize("gain_db", [-12.0, -6.0, -3.0, 3.0, 6.0])
def test_gain_changes_rms_by_exactly_the_requested_amount(
    tone: np.ndarray, gain_db: float
) -> None:
    result = build(f"gain:gain_db={gain_db}").apply(tone, SAMPLE_RATE)
    measured = 20 * np.log10(rms(result.audio) / rms(tone))

    assert measured == pytest.approx(gain_db, abs=1e-6)
    assert result.metadata["measured_gain_db"] == pytest.approx(gain_db, abs=1e-6)


def test_gain_does_not_normalise_away_its_own_effect(tone: np.ndarray) -> None:
    loud = build("gain:gain_db=20").apply(tone, SAMPLE_RATE)
    assert np.max(np.abs(loud.audio)) > 1.0
    assert loud.metadata["normalized"] is False
    assert loud.metadata["samples_over_full_scale"] > 0


def test_clipping_percentile_mode_clips_the_stated_share(tone: np.ndarray) -> None:
    result = build("clipping:threshold=99.0,mode=percentile").apply(tone, SAMPLE_RATE)
    assert result.metadata["clipped_fraction"] == pytest.approx(0.01, abs=0.002)


def test_clipping_rejects_an_unknown_mode(tone: np.ndarray) -> None:
    with pytest.raises(AttackError):
        build("clipping:threshold=0.5,mode=nonsense").apply(tone, SAMPLE_RATE)


# -------------------------------------------------------------- filtering


@pytest.mark.parametrize("cutoff", [4000.0, 2400.0, 1200.0])
def test_low_pass_attenuates_above_the_cutoff(noise_signal: np.ndarray, cutoff: float) -> None:
    result = build(f"low_pass:cutoff_hz={cutoff}").apply(noise_signal, SAMPLE_RATE)
    before = band_energy(noise_signal, cutoff * 1.5, SAMPLE_RATE / 2)
    after = band_energy(result.audio, cutoff * 1.5, SAMPLE_RATE / 2)

    assert 10 * np.log10(after / before) < -40.0
    # The passband survives.
    assert band_energy(result.audio, 0, cutoff * 0.5) == pytest.approx(
        band_energy(noise_signal, 0, cutoff * 0.5), rel=0.1
    )


def test_high_pass_attenuates_below_the_cutoff(noise_signal: np.ndarray) -> None:
    result = build("high_pass:cutoff_hz=1000").apply(noise_signal, SAMPLE_RATE)
    before = band_energy(noise_signal, 0, 500)
    after = band_energy(result.audio, 0, 500)
    assert 10 * np.log10(after / before) < -40.0


def test_notch_removes_its_band_and_leaves_the_rest(tone: np.ndarray) -> None:
    """The old frequency_filter compared bin frequencies for float equality
    and usually removed nothing at all; a notch must actually attenuate."""
    result = build("notch:center_hz=1000,quality=30").apply(tone, SAMPLE_RATE)

    removed = band_energy(result.audio, 950, 1050) / band_energy(tone, 950, 1050)
    kept = band_energy(result.audio, 2900, 3100) / band_energy(tone, 2900, 3100)

    assert 10 * np.log10(removed) < -20.0
    assert kept == pytest.approx(1.0, abs=0.05)


def test_notch_depth_is_honoured(tone: np.ndarray) -> None:
    result = build("notch:center_hz=1000,quality=30,depth_db=6").apply(tone, SAMPLE_RATE)
    removed = band_energy(result.audio, 995, 1005) / band_energy(tone, 995, 1005)
    assert 10 * np.log10(removed) == pytest.approx(-6.0, abs=1.5)


def test_filters_reject_a_cutoff_above_nyquist(tone: np.ndarray) -> None:
    """An 18 kHz low-pass is meaningless for a 16 kHz signal."""
    with pytest.raises(AttackError, match="Nyquist"):
        build("low_pass:cutoff_hz=18000").apply(tone, SAMPLE_RATE)


def test_smoothing_has_its_first_null_where_the_theory_says(noise_signal: np.ndarray) -> None:
    result = build("smoothing:window_length=8").apply(noise_signal, SAMPLE_RATE)
    null = result.metadata["first_null_hz"]
    assert null == pytest.approx(SAMPLE_RATE / 8)

    attenuation = band_energy(result.audio, null * 0.9, null * 1.1) / band_energy(
        noise_signal, null * 0.9, null * 1.1
    )
    assert 10 * np.log10(attenuation) < -10.0


# ------------------------------------------------------------- resampling


@pytest.mark.parametrize("intermediate", [8000, 22050, 48000])
def test_resample_round_trip_returns_the_original_rate_and_length(
    tone: np.ndarray, intermediate: int
) -> None:
    result = build(f"resample:intermediate_hz={intermediate}").apply(tone, SAMPLE_RATE)

    assert result.sample_rate == SAMPLE_RATE
    assert len(result.audio) == len(tone)
    assert result.metadata["intermediate_sample_rate"] == intermediate


def test_downsampling_round_trip_removes_content_above_the_new_nyquist(
    noise_signal: np.ndarray,
) -> None:
    result = build("resample:intermediate_hz=8000").apply(noise_signal, SAMPLE_RATE)
    above = band_energy(result.audio, 4400, SAMPLE_RATE / 2) / band_energy(
        noise_signal, 4400, SAMPLE_RATE / 2
    )
    assert 10 * np.log10(above) < -20.0


def test_upsampling_round_trip_is_nearly_lossless(tone: np.ndarray) -> None:
    result = build("resample:intermediate_hz=48000").apply(tone, SAMPLE_RATE)
    assert measured_snr_db(tone, result.audio) > 40.0


def test_clock_drift_changes_length_in_proportion(tone: np.ndarray) -> None:
    result = build("clock_drift:offset_ppm=1000").apply(tone, SAMPLE_RATE)
    expected = round(len(tone) * (1 + 1000 / 1e6))
    assert len(result.audio) == pytest.approx(expected, abs=2)


# ------------------------------------------------------------ quantization


@pytest.mark.parametrize("bits", [16, 12, 8, 4])
def test_bit_depth_uses_at_most_the_available_levels(tone: np.ndarray, bits: int) -> None:
    result = build(f"bit_depth:bits={bits}").apply(tone, SAMPLE_RATE)

    assert result.metadata["levels"] == 2 ** bits
    assert result.metadata["distinct_levels_used"] <= 2 ** bits
    # Every sample sits on the quantisation grid.
    step = result.metadata["step"]
    assert np.allclose(result.audio / step, np.rint(result.audio / step), atol=1e-9)


def test_bit_depth_error_is_bounded_by_half_a_step(tone: np.ndarray) -> None:
    result = build("bit_depth:bits=8").apply(tone, SAMPLE_RATE)
    assert result.metadata["max_absolute_error"] <= result.metadata["step"] / 2 + 1e-12


def test_bit_depth_dither_is_seeded(tone: np.ndarray) -> None:
    first = build("bit_depth:bits=8,dither=True,seed=3").apply(tone, SAMPLE_RATE).audio
    again = build("bit_depth:bits=8,dither=True,seed=3").apply(tone, SAMPLE_RATE).audio
    np.testing.assert_array_equal(first, again)


# --------------------------------------------------------------- temporal


@pytest.mark.parametrize("shift_ms", [1.0, 10.0, 100.0])
def test_time_shift_displaces_by_exactly_the_requested_amount(
    tone: np.ndarray, shift_ms: float
) -> None:
    result = build(f"time_shift:shift_ms={shift_ms}").apply(tone, SAMPLE_RATE)
    shift = int(round(shift_ms * SAMPLE_RATE / 1000))

    assert result.metadata["shift_samples"] == shift
    assert len(result.audio) == len(tone)
    np.testing.assert_allclose(result.audio[shift:shift + 500], tone[:500], atol=1e-9)
    np.testing.assert_allclose(result.audio[:shift], 0.0, atol=1e-12)


def test_circular_shift_keeps_every_sample(tone: np.ndarray) -> None:
    result = build("time_shift:shift_samples=100,mode=circular").apply(tone, SAMPLE_RATE)
    assert result.metadata["samples_lost"] == 0
    np.testing.assert_allclose(np.sort(result.audio), np.sort(tone), atol=1e-12)


@pytest.mark.parametrize("fraction", [0.001, 0.01, 0.05])
def test_crop_removes_the_exact_sample_count(tone: np.ndarray, fraction: float) -> None:
    result = build(f"crop:fraction={fraction}").apply(tone, SAMPLE_RATE)
    removed = int(round(len(tone) * fraction))

    assert len(result.audio) == len(tone) - removed
    assert result.metadata["removed_samples"] == removed


def test_zero_padding_offsets_the_content(tone: np.ndarray) -> None:
    result = build("zero_padding:fraction=0.1").apply(tone, SAMPLE_RATE)
    pad = result.metadata["padded_samples"]

    assert len(result.audio) == len(tone) + pad
    np.testing.assert_allclose(result.audio[:pad], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.audio[pad:pad + 500], tone[:500], atol=1e-9)


def test_sample_jitter_changes_length_by_the_expected_amount(tone: np.ndarray) -> None:
    deleted = build("sample_jitter:events=5,run_length=10,operation=delete").apply(tone, SAMPLE_RATE)
    inserted = build("sample_jitter:events=5,run_length=10,operation=insert").apply(tone, SAMPLE_RATE)

    assert deleted.metadata["length_delta"] == -50
    assert inserted.metadata["length_delta"] == 50


def test_dropout_zeroes_runs_without_changing_length(tone: np.ndarray) -> None:
    result = build("dropout:fraction=0.01,run_length=20").apply(tone, SAMPLE_RATE)

    assert len(result.audio) == len(tone)
    assert result.metadata["zeroed_samples"] > 0


def test_speed_and_time_stretch_differ_in_pitch(tone: np.ndarray) -> None:
    """Both shorten the signal; only the speed change moves the pitch."""
    speed = build("speed:rate=1.1").apply(tone, SAMPLE_RATE)
    stretch = build("time_stretch:rate=1.1").apply(tone, SAMPLE_RATE)

    def dominant(signal: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(signal))
        return float(np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)[int(np.argmax(spectrum))])

    assert dominant(speed.audio) == pytest.approx(1100, rel=0.05)
    assert dominant(stretch.audio) == pytest.approx(1000, rel=0.05)
    assert speed.metadata["pitch_preserved"] is False
    assert stretch.metadata["pitch_preserved"] is True


def test_pitch_shift_moves_the_tone_and_keeps_the_duration(tone: np.ndarray) -> None:
    result = build("pitch_shift:semitones=12").apply(tone, SAMPLE_RATE)
    spectrum = np.abs(np.fft.rfft(result.audio))
    dominant = float(np.fft.rfftfreq(len(result.audio), 1 / SAMPLE_RATE)[int(np.argmax(spectrum))])

    assert dominant == pytest.approx(2000, rel=0.05)
    assert len(result.audio) == pytest.approx(len(tone), rel=0.01)


# --------------------------------------------------------------- acoustic


def test_echo_puts_a_cepstral_peak_at_its_delay(noise_signal: np.ndarray) -> None:
    result = build("echo:delay_ms=25,attenuation=0.5").apply(noise_signal, SAMPLE_RATE)
    cepstrum = np.real(np.fft.irfft(np.log(np.abs(np.fft.rfft(result.audio)) + 1e-12)))

    delay = int(0.025 * SAMPLE_RATE)
    neighbourhood = np.abs(np.concatenate([cepstrum[delay - 60:delay - 5], cepstrum[delay + 5:delay + 60]]))
    assert cepstrum[delay] > 5 * float(np.median(neighbourhood))


def test_reverberation_is_seeded_and_keeps_the_length(tone: np.ndarray) -> None:
    first = build("reverb:rt60_seconds=0.3,seed=2").apply(tone, SAMPLE_RATE)
    again = build("reverb:rt60_seconds=0.3,seed=2").apply(tone, SAMPLE_RATE)

    np.testing.assert_allclose(first.audio, again.audio)
    assert len(first.audio) == len(tone)


def test_acoustic_channel_records_every_stage(tone: np.ndarray) -> None:
    result = build("acoustic_channel").apply(tone, SAMPLE_RATE)
    stages = [stage["stage"] for stage in result.metadata["stages"]]

    assert stages == ["band_pass", "reverb", "awgn", "gain"]
    assert result.metadata["simulated"] is True


# ------------------------------------------------------------------ codec


@needs_ffmpeg
@pytest.mark.parametrize("codec,bitrate", [("mp3", 128), ("aac", 96), ("opus", 32)])
def test_codec_really_encodes(tone: np.ndarray, codec: str, bitrate: int) -> None:
    result = build(f"{codec}:bitrate_kbps={bitrate}").apply(tone, SAMPLE_RATE)
    metadata = result.metadata

    assert metadata["codec"] == codec
    assert metadata["bitrate_kbps"] == bitrate
    # A real compressed file was produced, substantially smaller than PCM.
    assert metadata["compressed_bytes"] > 0
    assert metadata["compression_ratio"] > 1.5
    assert "ffmpeg" in metadata["ffmpeg_version"].lower()
    # And the audio came back changed but recognisable.
    assert len(result.audio) == len(tone)
    assert 0.0 < measured_snr_db(tone, result.audio) < 80.0


@needs_ffmpeg
def test_lower_bitrate_damages_more(noise_signal: np.ndarray) -> None:
    high = build("mp3:bitrate_kbps=192").apply(noise_signal, SAMPLE_RATE)
    low = build("mp3:bitrate_kbps=64").apply(noise_signal, SAMPLE_RATE)
    assert low.metadata["compressed_bytes"] < high.metadata["compressed_bytes"]


def test_codec_rejects_an_unknown_name(tone: np.ndarray) -> None:
    with pytest.raises(AttackError):
        build("codec:codec=flac1234").apply(tone, SAMPLE_RATE)


# --------------------------------------------------------------- pipeline


def test_pipeline_applies_stages_in_order_and_records_each(tone: np.ndarray) -> None:
    result = build("pipeline:name=broadcast").apply(tone, SAMPLE_RATE)
    stages = [stage["attack"] for stage in result.metadata["stage_metadata"]]

    assert stages == ["compression_dynamic", "resample", "awgn"]
    assert result.metadata["label"] == "broadcast"


# -------------------------------------------------- input handling contract


def test_stereo_is_processed_per_channel_without_downmixing() -> None:
    rng = np.random.default_rng(5)
    stereo = rng.normal(0, 0.1, (SAMPLE_RATE, 2))
    result = build("low_pass:cutoff_hz=2000").apply(stereo, SAMPLE_RATE)

    assert result.audio.shape == stereo.shape
    assert result.metadata["channels"] == 2
    assert not np.allclose(result.audio[:, 0], result.audio[:, 1])


def test_integer_input_keeps_its_dtype_and_is_scaled_not_reinterpreted() -> None:
    rng = np.random.default_rng(6)
    pcm = (rng.normal(0, 0.1, SAMPLE_RATE) * 32767).astype(np.int16)
    result = build("gain:gain_db=-6").apply(pcm, SAMPLE_RATE)

    assert result.audio.dtype == np.int16
    assert result.metadata["input_dtype"] == "int16"
    assert rms(result.audio.astype(np.float64)) == pytest.approx(
        rms(pcm.astype(np.float64)) * 10 ** (-6 / 20), rel=0.01
    )


def test_non_finite_input_is_rejected() -> None:
    broken = np.zeros(1000)
    broken[10] = np.nan
    with pytest.raises(AttackError):
        build("gain:gain_db=-3").apply(broken, SAMPLE_RATE)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(AttackError):
        build("gain:gain_db=-3").apply(np.array([]), SAMPLE_RATE)


def test_short_signal_is_handled_or_refused_clearly(tone: np.ndarray) -> None:
    short = tone[:64]
    # A 100 ms shift cannot be applied to a 4 ms signal, and says so.
    with pytest.raises(AttackError):
        build("time_shift:shift_ms=100").apply(short, SAMPLE_RATE)
    # A gain change is always fine.
    assert len(build("gain:gain_db=-3").apply(short, SAMPLE_RATE).audio) == 64


def test_metadata_always_describes_the_transformation(tone: np.ndarray) -> None:
    result = build("awgn:snr_db=25,seed=11").apply(tone, SAMPLE_RATE)
    metadata = result.metadata

    assert metadata["attack"] == "awgn"
    assert metadata["category"] == "noise"
    assert metadata["parameters"] == {"snr_db": 25.0, "seed": 11, "prevent_clipping": False}
    assert metadata["input_sample_rate"] == SAMPLE_RATE
    assert metadata["output_sample_rate"] == SAMPLE_RATE
    assert metadata["input_length"] == len(tone)
    assert metadata["output_length"] == len(tone)


# ------------------------------------------------------ presets and registry


@pytest.mark.parametrize("preset,minimum", [("quick", 5), ("standard", 40), ("full", 100)])
def test_benchmark_suites_are_populated_and_buildable(preset: str, minimum: int) -> None:
    specs = benchmark_suite(preset, SAMPLE_RATE)
    assert len(specs) >= minimum
    for spec in specs:
        build(spec, sample_rate=SAMPLE_RATE)


def test_severity_maps_to_explicit_numbers() -> None:
    values = [severity_parameters("awgn", level)["snr_db"] for level in Severity]
    assert values == sorted(values, reverse=True)  # harsher levels mean lower SNR


def test_severity_respects_the_sampling_rate() -> None:
    """A cutoff or resampling target valid at 44.1 kHz may be nonsense at 16 kHz."""
    for level in Severity:
        narrowband = severity_parameters("low_pass", level, 16000)["cutoff_hz"]
        wideband = severity_parameters("low_pass", level, 44100)["cutoff_hz"]
        assert narrowband < 8000
        assert wideband < 22050

        target = severity_parameters("resample", level, 16000)["intermediate_hz"]
        assert target in {8000, 22050}


def test_legacy_attack_names_still_build() -> None:
    assert build("additive_noise").name == "awgn"
    assert build("amplitude_scaling").name == "gain"
    assert build("frequency_filter").name == "notch"
    assert build("mp3_compression").parameters()["codec"] == "mp3"


def test_unknown_specification_is_rejected() -> None:
    with pytest.raises(AttackError):
        build("not_a_real_attack")
    with pytest.raises(AttackError):
        build("awgn:snr_db")
    with pytest.raises(AttackError):
        build("awgn@ferocious")
