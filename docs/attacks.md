# Attacks and channel models

An attack transforms the stego waveform before extraction. The registry contains **26 attack classes**
and **four codec shortcuts** (`mp3`, `aac`, `opus`, `vorbis`); these names do not represent 30 independent algorithms.
Five named pipelines combine operations in a fixed order. Legacy aliases resolve to canonical names.

AudioMarkBench distinguishes common processing from detector-aware removal and forgery attacks
([§3 and Appendix A.2](https://arxiv.org/html/2406.06979v2)). TAF’s catalogue below models signal processing
and channel impairments; it does not implement that paper’s complete adversarial benchmark.
The descriptions and parameters refer to the local implementation, not to claimed paper results.

| Name | Transformation | Main controls |
| --- | --- | --- |
| `awgn` | Adds white Gaussian noise at a specified signal-to-noise ratio. | `snr_db` |
| `pink_noise` | Adds coloured noise with approximately 1/f power spectrum. | `snr_db` |
| `impulse_noise` | Adds sparse impulses to model clicks or transient corruption. | `snr_db, density` |
| `codec` | General FFmpeg encode/decode round trip; choose the codec explicitly. | `codec, bitrate_kbps` |
| `mp3` | MP3 perceptual coding round trip. | `bitrate_kbps` |
| `aac` | AAC perceptual coding round trip. | `bitrate_kbps` |
| `opus` | Opus coding round trip for speech/audio transmission. | `bitrate_kbps` |
| `vorbis` | Vorbis perceptual coding round trip. | `bitrate_kbps` |
| `low_pass` | Suppresses frequencies above a cutoff. | `cutoff_hz, order` |
| `high_pass` | Suppresses frequencies below a cutoff. | `cutoff_hz, order` |
| `band_pass` | Retains a bounded frequency range. | `low_hz, high_hz` |
| `notch` | Attenuates a narrow band around a selected frequency. | `center_hz, quality` |
| `smoothing` | Applies a moving-average filter, suppressing rapid sample changes. | `window_length` |
| `resample` | Converts to an intermediate sampling rate and back to the source rate. | `intermediate_hz` |
| `clock_drift` | Simulates sampling-clock mismatch through a small rate offset. | `offset_ppm` |
| `bit_depth` | Quantises sample amplitudes to a lower bit depth, optionally with dither. | `bits, dither` |
| `gain` | Applies a global level change in decibels. | `gain_db` |
| `clipping` | Limits waveform peaks, introducing nonlinear distortion. | `threshold, mode` |
| `compression_dynamic` | Applies memoryless amplitude compression above a linear threshold, then restores the original peak. | `threshold, ratio` |
| `time_shift` | Shifts samples with zero padding by default; circular rotation is optional. | `shift_ms, shift_samples, mode` |
| `crop` | Removes a fraction of the signal at a selected position. | `fraction, position` |
| `zero_padding` | Adds silence at a selected boundary. | `fraction, position` |
| `sample_jitter` | Inserts or deletes individual samples, disrupting synchronisation. | `events` |
| `dropout` | Zeros sample runs to simulate missing observations. | `fraction, run_length` |
| `time_stretch` | Changes duration with approximate pitch preservation. | `rate` |
| `speed` | Changes playback speed, affecting duration and pitch together. | `rate` |
| `pitch_shift` | Changes pitch while approximately preserving duration. | `semitones` |
| `echo` | Adds one delayed, attenuated copy of the signal. | `delay_ms, attenuation` |
| `reverb` | Convolves audio with a synthetic decaying room response. | `rt60_seconds` |
| `acoustic_channel` | Combines simulated room, bandwidth, noise and optional clock effects. | `rt60_seconds, snr_db, clock_offset_ppm` |

## Composite channels

These are TAF-defined scenarios, not standardised models of particular services.

| Pipeline | Processing order |
| --- | --- |
| `streaming_upload` | AAC → dynamic compression → MP3 |
| `voice_call` | Speech-band filtering → Opus → sample dropouts |
| `broadcast` | Dynamic compression → resampling round trip → white noise |
| `over_the_air` | Simulated acoustic channel → AAC |
| `desync_attack` | Time shift → speed change → MP3 |

Use `pipeline:name=voice_call` in an experiment. Simulated reverberation and playback do not substitute
for physical loudspeaker–microphone measurements; echo and re-recording studies motivate these tests
([32](references.md#ref-32), [33](references.md#ref-33), [42](references.md#ref-42)).
All stochastic operations accept a seed. Experiment exports record resolved parameters and seeds.

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


See the [historical audit](attack-history.md) for changes to legacy attacks.
