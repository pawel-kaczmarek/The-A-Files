# Robustness attacks

This document describes the attack framework used to evaluate how well an
embedded payload survives realistic audio processing: what each attack models,
how it is parameterised, why those parameter ranges were chosen, and what the
previous implementation got wrong.

- [Threat model](#threat-model)
- [Audit of the previous attacks](#audit-of-the-previous-attacks)
- [Taxonomy](#taxonomy)
- [Parameters and severity](#parameters-and-severity)
- [Benchmark suites](#benchmark-suites)
- [Which attacks matter for which method](#which-attacks-matter-for-which-method)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)

## Threat model

An attack is any transformation a stego signal may undergo between embedding
and extraction:

- **incidental processing** — level changes, filtering, dynamic-range control;
- **storage and transmission** — lossy codecs, bit-depth reduction, packet loss;
- **format conversion** — sample-rate conversion, container changes;
- **playback and recapture** — loudspeaker, room, microphone, clock mismatch;
- **deliberate removal** — an adversary who wants the payload gone while the
  audio stays usable.

A benchmark is only meaningful if each transformation is what it claims to be,
if its severity is controlled by a number that means the same thing on every
file, and if the whole run can be repeated exactly.

## Audit of the previous attacks

Every attack in the previous implementation was inspected mathematically and
measured on VCTK speech at 16 kHz. The table records what was found and what
replaced it.

| Old attack | What it actually did | Problem | Replacement | Parameters |
| --- | --- | --- | --- | --- |
| `additive_noise` | `np.random.normal(0, std)` added, `std=0.001` absolute, global RNG | Severity depended on recording level, so the same setting was different attacks on different files; no seed, so runs were not reproducible | `awgn`, plus separate `pink_noise` and `impulse_noise` | `snr_db`, `seed`, `prevent_clipping` |
| `frequency_filter` | Zeroed FFT bins where `np.abs(W) == cutoff` — **exact float equality** | Measured: on a 12345-sample signal this removes **zero** bins; on a 32000-sample signal exactly 2 of 32000. The attack was essentially a no-op and modelled nothing | `notch` (second-order IIR notch) | `center_hz`, `quality`, `depth_db` |
| `resample` | Resampled to 27500 Hz and **left the signal there** | The decoder received audio at a rate it was never built for, so the experiment measured API tolerance, not robustness to conversion | `resample` (round trip back to the source rate) | `intermediate_hz`, `restore_length` |
| `low_pass_filter` | Butterworth order 16, fixed 2000 Hz, causal `sosfilt` | Cutoff not validated against Nyquist (meaningless at low rates); order 16 is not plausible audio processing; causal filtering added an undocumented group delay that desynchronises decoders | `low_pass` | `cutoff_hz`, `order` (6), `zero_phase` |
| `amplitude_scaling` | `samples * 1.1` | Linear factor rather than the decibels used by every volume control and every paper; clipping neither prevented nor reported | `gain` | `gain_db`, `prevent_clipping` |
| `flip_random_samples` | Sign-inverted 200 random samples, global RNG | Models no physical channel; unseeded | `impulse_noise` (SNR-controlled sparse impulses) | `snr_db`, `density`, `seed` |
| `cut_random_samples` | Zeroed 200 random samples, global RNG | Reasonable phenomenon (dropouts) but unseeded and specified as a raw count rather than a fraction | `dropout` | `fraction`, `run_length`, `seed` |
| `sample_suppression` | Zeroed random runs, global RNG | Duplicate of `cut_random_samples` under another name | merged into `dropout` | as above |
| `time_stretch` | `rate=2.0` (phase vocoder) | Doubling the tempo destroys every method for the trivial reason that half the signal is gone; the informative range is a few percent | `time_stretch` | `rate` (0.95–1.05) |
| `pitch_shift` | 4 semitones | Grossly audible, so not a realistic covert attack | `pitch_shift` | `semitones` (0.25–2) |
| `quantization` | `round(x * 2^(b-1)) / 2^(b-1)` | Roughly right but the quantiser was implicit: no stated grid convention, no dither option, and the level count was off by a factor of two from the nominal depth | `bit_depth` | `bits`, `dither`, `mode`, `seed` |
| `echo_addition` | `x[n] + decay*x[n-D]`, `delay=0.1 s` | Model correct; delay at the far end of the plausible range and specified in seconds | `echo` | `delay_ms`, `attenuation` |
| `smoothing` | Moving average, `mode="same"` | Correct, but causal/zero-phase not stated and the nulls of the boxcar response were undocumented | `smoothing` | `window_length`, `zero_phase` |
| `crop` | Removed a fraction split across both ends | Reasonable, but the definition was implicit and could not be varied | `crop` | `fraction`, `position`, `seed` |
| `zero_padding` | Prepended zeros | Correct | `zero_padding` | `fraction`, `position` |
| `speed_change` | Resample without pitch correction | Correct, and correctly distinguished from `time_stretch` | `speed` | `rate` |
| `mp3/aac/opus_compression` | Real FFmpeg round trip | Correct in principle; bitrate passed as a string, no codec provenance recorded, no encoder-delay handling | `mp3`, `aac`, `opus`, `vorbis` | `bitrate_kbps`, `align_delay`, `restore_length` |

Cross-cutting problems, all now fixed: no attack recorded its parameters; the
benchmark applied every attack with its defaults and had no way to sweep;
stochastic attacks used the global RNG; nothing validated parameters against
the sample rate; and discovery by reflection over `CorruptedWavFile` had begun
listing helper methods as attacks.

**Added**, because the taxonomy had visible gaps: `pink_noise`, `impulse_noise`,
`band_pass`, `high_pass` (parameterised), `clock_drift`, `clipping`,
`compression_dynamic`, `time_shift`, `sample_jitter`, `reverb`,
`acoustic_channel`, `vorbis`, and composed `pipeline` channels.

## Taxonomy

| Family | Attacks | Models |
| --- | --- | --- |
| Codec | `mp3`, `aac`, `opus`, `vorbis` | Distribution through perceptual coding |
| Noise | `awgn`, `pink_noise`, `impulse_noise` | Channel noise, ambient noise, clicks and bit errors |
| Filtering | `low_pass`, `high_pass`, `band_pass`, `notch`, `smoothing` | Band limitation, transmission channels, tone removal |
| Resampling | `resample`, `clock_drift` | Format conversion, device clock mismatch |
| Quantization | `bit_depth` | Lower-depth storage and conversion |
| Amplitude | `gain`, `clipping`, `compression_dynamic` | Level changes, headroom loss, loudness processing |
| Temporal | `time_shift`, `crop`, `zero_padding`, `sample_jitter`, `dropout`, `time_stretch`, `speed`, `pitch_shift` | Editing, packet loss, tempo and speed changes |
| Acoustic | `echo`, `reverb`, `acoustic_channel` | Reflections, rooms, playback and recapture |
| Pipeline | `streaming_upload`, `voice_call`, `broadcast`, `over_the_air`, `desync_attack` | Real multi-stage distribution chains |

## Parameters and severity

Severity labels are never parameters. `MILD`, `MODERATE`, `STRONG` and
`EXTREME` resolve through `taf.attacks.presets` into explicit numbers that are
written into every result row, so a result is reproducible without this
document.

| Attack | MILD | MODERATE | STRONG | EXTREME |
| --- | --- | --- | --- | --- |
| `awgn`, `pink_noise` (SNR dB) | 40 | 25 | 15 | 5 |
| `mp3` (kbps) | 256 | 128 | 96 | 64 |
| `aac` (kbps) | 192 | 128 | 96 | 64 |
| `opus` (kbps) | 128 | 96 | 64 | 32 |
| `low_pass` (× Nyquist) | 0.90 | 0.50 | 0.30 | 0.15 |
| `high_pass` (Hz) | 20 | 100 | 300 | 1000 |
| `bit_depth` (bits) | 16 | 12 | 10 | 8 |
| `gain` (dB) | −3 | −6 | −12 | +6 |
| `crop` (fraction) | 0.001 | 0.01 | 0.05 | 0.10 |
| `time_shift` (ms) | 1 | 10 | 100 | 250 |
| `time_stretch`, `speed` (rate) | 1.01 | 0.99 | 1.05 | 0.95 |
| `pitch_shift` (semitones) | 0.25 | 0.5 | 1.0 | 2.0 |
| `echo` (ms, attenuation) | 5, 0.1 | 25, 0.25 | 50, 0.5 | 100, 0.5 |
| `reverb` (RT60 s) | 0.2 | 0.5 | 1.0 | 1.5 |
| `clipping` (peak fraction) | 0.95 | 0.90 | 0.80 | 0.70 |
| `clock_drift` (ppm) | 10 | 50 | 100 | 500 |
| `resample` (intermediate) | one step up | one step down | two steps down | lowest available |

### Why these ranges

These are engineering benchmark choices, informed by common practice in the
audio watermarking literature. They are stated with their reasoning rather than
attributed to any particular publication.

- **SNR 40 → 5 dB.** 40 dB is at the edge of audibility on speech; 25 dB is
  clearly audible but undamaging; 15 dB is a poor channel; 5 dB is noise
  comparable to the signal, included to locate where a method finally fails.
- **Bitrates.** The ladder spans transparent (256/192 kbps) to the point where
  coding artefacts are plainly audible (64/32 kbps), which is the range in
  which distributed audio actually exists.
- **Filter cutoffs as a fraction of Nyquist.** A cutoff fixed in Hertz is
  wrong at some sampling rate: an 18 kHz low-pass does nothing to 16 kHz
  speech. Expressing it relative to Nyquist keeps the attack meaningful at any
  rate. High-pass corners stay absolute because the low end does not move with
  the sampling rate.
- **Tempo and speed of ±1–5%.** Deviations of a percent or two are inaudible
  to most listeners and accumulate into a large sample offset across a clip,
  which is what makes them a realistic covert attack. Larger factors are
  audible and therefore less interesting.
- **Time shifts from one millisecond.** One millisecond is 16 samples at
  16 kHz — inaudible, and already fatal to a decoder that indexes by position.
- **Echo delays of 5–100 ms.** Below roughly 10 ms the echo fuses with the
  direct sound and colours it; above 50 ms it is heard as a repetition.
- **RT60 of 0.2–1.5 s** spans a treated room, a living room, a hall and a
  large hall.
- **Clock drift of 10–500 ppm.** Consumer converters routinely differ by tens
  to hundreds of ppm.

## Benchmark suites

| Preset | Size | Purpose |
| --- | --- | --- |
| `quick` | 9 configurations | Smoke test: one attack per family. Not for drawing conclusions. |
| `standard` | ~51 configurations | Default scientific benchmark: every family at several levels. |
| `full` | ~168 configurations | Dense sweep: every attack at every severity, plus the pipelines. |

```python
from taf.attacks.presets import benchmark_suite

benchmark_suite("standard", sample_rate=16000)
```

Or from an experiment configuration:

```yaml
experiment_type: attack_robustness
attack_preset: standard
```

## Which attacks matter for which method

Embedding domain determines vulnerability. This is a guide to reading results,
not a claim about any particular implementation.

| Method family | Especially vulnerable to | Usually survives |
| --- | --- | --- |
| Sample-domain LSB (`LsbMethod`, `FBSMethod`, `PrimeFactorInterpolated`, `WirelessDwtLsb`, `AacStc`) | `bit_depth`, any codec, `awgn`, `gain`, `dropout` | Nothing much; these are steganography, not watermarking |
| Transform-domain quantisation (`DctB1`, `BlindSvd`, `Qim`, `DwtLsb`) | Codecs, `awgn`, `bit_depth`; `gain` when the step is absolute rather than signal-derived | Filtering inside the embedded band, mild resampling |
| Amplitude- and norm-relation (`NormSpace`, `DctDeltaLsb`, `LowFrequencyAmplitude`, `Lwt`) | `compression_dynamic`, `clipping`, heavy filtering of the carrier band | `gain`, mild noise, mild codecs |
| Spread spectrum (`Dsss`, `ImprovedSpreadSpectrum`, `LearnableEmbeddingGa`) | `low_pass` / `band_pass` removing the chip band, low-bitrate codecs, `time_shift` | `gain`, moderate `awgn`, `bit_depth` |
| Echo hiding (`Echo`, `BackwardForwardEcho`, `TimeSpreadEcho`) | `echo` (a competing cepstral peak), `reverb`, `smoothing`, resampling | `gain`, moderate noise |
| Phase (`PhaseCoding`, `ImprovedPhaseCoding`) | Codecs (phase is not preserved), filtering, resampling | `gain` |
| Histogram (`Histogram`) | Phase-vocoder `time_stretch`, heavy noise | `crop`, `speed`, `gain`, filtering — the point of the method |
| Neural (`AudioSeal`, `WavMark`) | Designed against most of this set; test with pipelines and `acoustic_channel` | Most single attacks |
| Synchronisation-sensitive (nearly all of the above) | `time_shift`, `crop`, `zero_padding`, `sample_jitter`, `speed`, `clock_drift` | — |

## Usage

```python
from taf.attacks import build

attack = build("awgn:snr_db=20,seed=7")
result = attack.apply(samples, sample_rate)

result.audio                      # processed signal, original dtype
result.metadata["parameters"]     # {"snr_db": 20.0, "seed": 7, ...}
result.metadata["measured_snr_db"]
```

Specification strings accept parameters (`"awgn:snr_db=20"`), severities
(`"mp3@strong"`) and pipelines (`"pipeline:name=voice_call"`). The chainable
builder is still available:

```python
from taf.attacks import CorruptedWavFile

corrupted = CorruptedWavFile(wav_file).gain(-6).additive_noise(snr_db=25).mp3_compression(128)
corrupted.metadata          # one record per stage
```

## Reproducibility

Every result row records the attack name, its resolved parameters, the seed,
input and output sample rate, input and output length, dtype conversion, any
clipping or length correction, and — for codecs — the encoder and the FFmpeg
version. Repeating a run with the same input, configuration and seed produces
identical audio; this is asserted in `tests/test_attacks_dsp.py`.

Inside an experiment the seed written in a specification is only a default.
Unless the specification pins one (`"awgn:seed=7"`), the evaluation replaces
it with a seed derived from the experiment seed, the file, the repetition and
the attack (`taf.evaluation.seeding`, applied with `registry.reseed`). Each
stage of a pipeline gets its own child seed. As a result:

- repetitions and files sample the channel independently, so the spread of
  the results includes the randomness of the attack; with a fixed seed every
  repetition used to see the same noise realisation;
- every method meets the same realisation in a given trial (common random
  numbers), which keeps the comparison between methods paired and fair.

## Limitations

- **Codec attacks require FFmpeg** and their results depend on the build and
  encoder versions, which are recorded but not controlled. Without FFmpeg
  these attacks raise `AttackToolUnavailableError` and the tests skip.
- **The acoustic channel is a simulation**, not a recording. It composes band
  limitation, a synthetic room response, noise, gain and optional clock drift.
  Real capture also involves microphone nonlinearity, room modes, movement and
  ambient events. Results from it should be reported as a simulated acoustic
  path.
- **Room impulse responses are synthetic** — exponentially decaying Gaussian
  noise with the requested RT60. This reproduces the decay and diffuse
  character of a room but not its modal structure or frequency-dependent
  absorption. No measured RIR corpus is bundled.
- **An upsampling round trip is not bit-exact.** The polyphase filter's
  transition band removes content just below Nyquist, so even the mildest
  resampling costs a little signal. The measured SNR is recorded per row.
- **Perceptual quality is measured with the packaged speech metrics** (PESQ,
  STOI and the rest), which are speech-specific. Applying them to music or
  general audio is not supported by their design and should be avoided.
- **Encoder delay is estimated by cross-correlation** and removed by default.
  The estimate is recorded; where a real channel would preserve that offset,
  set `align_delay=False`.
