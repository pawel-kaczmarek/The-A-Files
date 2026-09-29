# Attacks and channel models

An attack transforms the stego waveform before extraction. Five named pipelines combine operations in a
fixed order. Legacy aliases resolve to canonical names.

AudioMarkBench distinguishes common processing from detector-aware removal and forgery attacks
([§3 and Appendix A.2](https://arxiv.org/html/2406.06979v2)). TAF’s catalogue below models signal processing
and channel impairments; it does not implement that paper’s complete adversarial benchmark.
The descriptions and parameters refer to the local implementation, not to claimed paper results.

<!-- catalogue:attacks-table -->
The registry contains **26 attack classes** and **4 codec shortcuts** (`mp3`, `aac`, `opus`, `vorbis`); the shortcuts fix the `codec` parameter and are not separate algorithms.

| Attack | Transformation | Parameters |
| --- | --- | --- |
| [`awgn`](#awgn) | Adds white Gaussian noise to model a noisy recording or transmission channel. | `snr_db`, `prevent_clipping` |
| [`impulse_noise`](#impulse-noise) | Injects sparse strong impulses, resembling clicks or isolated transmission errors. | `snr_db`, `density`, `prevent_clipping` |
| [`pink_noise`](#pink-noise) | Adds pink noise, concentrating more disturbance at low frequencies than white noise. | `snr_db`, `prevent_clipping` |
| [`aac`](#codec) | Tests whether the message survives real AAC lossy encoding and decoding. | `bitrate_kbps`, `align_delay`, `restore_length` |
| [`codec`](#codec) | Tests whether the message survives real lossy encoding and decoding (MP3, AAC, Opus or Vorbis). | `codec`, `bitrate_kbps`, `align_delay`, `restore_length` |
| [`mp3`](#codec) | Tests whether the message survives real MP3 lossy encoding and decoding. | `bitrate_kbps`, `align_delay`, `restore_length` |
| [`opus`](#codec) | Tests whether the message survives real Opus lossy encoding and decoding. | `bitrate_kbps`, `align_delay`, `restore_length` |
| [`vorbis`](#codec) | Tests whether the message survives real Vorbis lossy encoding and decoding. | `bitrate_kbps`, `align_delay`, `restore_length` |
| [`band_pass`](#band-pass) | Keeps only a selected frequency band, modelling a bandwidth-limited channel. | `low_hz`, `high_hz`, `order`, `zero_phase` |
| [`high_pass`](#high-pass) | Attenuates low frequencies, as in rumble removal or an AC-coupled recording path. | `cutoff_hz`, `order`, `zero_phase` |
| [`low_pass`](#low-pass) | Attenuates high frequencies to test whether the watermark depends on the upper band. | `cutoff_hz`, `order`, `zero_phase` |
| [`notch`](#notch) | Suppresses a narrow frequency region, such as hum or a tonal interferer. | `center_hz`, `quality`, `depth_db` |
| [`smoothing`](#smoothing) | Smooths samples by replacing them with local moving averages. | `window_length`, `zero_phase` |
| [`clock_drift`](#clock-drift) | Simulates a small mismatch between playback and recording clocks. | `offset_ppm` |
| [`resample`](#resample) | Converts to an intermediate sample rate and back to test rate-conversion damage. | `intermediate_hz`, `restore_length` |
| [`bit_depth`](#bit-depth) | Requantises audio to fewer PCM amplitude levels. | `bits`, `dither`, `mode` |
| [`clipping`](#clipping) | Limits signal peaks, modelling saturation or overdriven audio. | `threshold`, `mode` |
| [`compression_dynamic`](#compression-dynamic) | Reduces the difference between loud and quiet samples, then restores the original peak. | `threshold`, `ratio` |
| [`gain`](#gain) | Changes playback level to test dependence on absolute amplitude. | `gain_db`, `prevent_clipping` |
| [`crop`](#crop) | Removes part of the recording to test payload loss and framing sensitivity. | `fraction`, `position` |
| [`dropout`](#dropout) | Replaces short audio regions with zeros, modelling missing packets or recording dropouts. | `fraction`, `run_length` |
| [`pitch_shift`](#pitch-shift) | Raises or lowers pitch while keeping approximately the same duration. | `semitones` |
| [`sample_jitter`](#sample-jitter) | Inserts or deletes short sample runs at scattered positions. | `events`, `run_length`, `operation` |
| [`speed`](#speed) | Changes playback speed, affecting both duration and pitch. | `rate` |
| [`time_shift`](#time-shift) | Moves the recording in time to test decoder synchronisation. | `shift_samples`, `shift_ms`, `mode` |
| [`time_stretch`](#time-stretch) | Changes duration while approximately preserving pitch. | `rate` |
| [`zero_padding`](#zero-padding) | Adds silence at the start, the end or both, without changing the retained samples. | `fraction`, `position` |
| [`acoustic_channel`](#acoustic-channel) | Combines several effects to approximate a loudspeaker-to-microphone channel. | `band_low_hz`, `band_high_hz`, `rt60_seconds`, `snr_db`, `gain_db`, `clock_offset_ppm` |
| [`echo`](#echo) | Adds one delayed, attenuated copy of the signal, modelling a single acoustic reflection. | `delay_ms`, `attenuation`, `prevent_clipping` |
| [`reverb`](#reverb) | Simulates reverberation by convolving audio with a generated room impulse response. | `rt60_seconds`, `mix`, `trim_to_input` |
<!-- /catalogue:attacks-table -->

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
document. The table lists the values resolved at 16 kHz; rate-dependent levels (filter
cutoffs, resampling targets) move with the sampling rate.

<!-- catalogue:attacks-severity -->
| Attack | MILD | MODERATE | STRONG | EXTREME |
| --- | --- | --- | --- | --- |
| `awgn` | snr_db=40.0 | snr_db=25.0 | snr_db=15.0 | snr_db=5.0 |
| `impulse_noise` | snr_db=40.0, density=0.001 | snr_db=25.0, density=0.005 | snr_db=15.0, density=0.01 | snr_db=5.0, density=0.05 |
| `pink_noise` | snr_db=40.0 | snr_db=25.0 | snr_db=15.0 | snr_db=5.0 |
| `aac` | bitrate_kbps=192 | bitrate_kbps=128 | bitrate_kbps=96 | bitrate_kbps=64 |
| `codec` | bitrate_kbps=160 | bitrate_kbps=128 | bitrate_kbps=96 | bitrate_kbps=64 |
| `mp3` | bitrate_kbps=160 | bitrate_kbps=128 | bitrate_kbps=96 | bitrate_kbps=64 |
| `opus` | bitrate_kbps=128 | bitrate_kbps=96 | bitrate_kbps=64 | bitrate_kbps=32 |
| `vorbis` | bitrate_kbps=96 | bitrate_kbps=64 | bitrate_kbps=48 | bitrate_kbps=32 |
| `band_pass` | low_hz=20.0, high_hz=7200.0 | low_hz=100.0, high_hz=4000.0 | low_hz=300.0, high_hz=2400.0 | low_hz=1000.0, high_hz=1200.0 |
| `high_pass` | cutoff_hz=20.0 | cutoff_hz=100.0 | cutoff_hz=300.0 | cutoff_hz=1000.0 |
| `low_pass` | cutoff_hz=7200.0 | cutoff_hz=4000.0 | cutoff_hz=2400.0 | cutoff_hz=1200.0 |
| `notch` | center_hz=2000.0, quality=60.0 | center_hz=2000.0, quality=30.0 | center_hz=2000.0, quality=10.0 | center_hz=2000.0, quality=3.0 |
| `smoothing` | window_length=3 | window_length=5 | window_length=9 | window_length=17 |
| `clock_drift` | offset_ppm=10.0 | offset_ppm=50.0 | offset_ppm=100.0 | offset_ppm=500.0 |
| `resample` | intermediate_hz=22050 | intermediate_hz=8000 | intermediate_hz=8000 | intermediate_hz=8000 |
| `bit_depth` | bits=16 | bits=12 | bits=10 | bits=8 |
| `clipping` | threshold=0.95, mode='peak' | threshold=0.9, mode='peak' | threshold=0.8, mode='peak' | threshold=0.7, mode='peak' |
| `compression_dynamic` | ratio=2.0 | ratio=4.0 | ratio=8.0 | ratio=16.0 |
| `gain` | gain_db=-3.0 | gain_db=-6.0 | gain_db=-12.0 | gain_db=6.0 |
| `crop` | fraction=0.001 | fraction=0.01 | fraction=0.05 | fraction=0.1 |
| `dropout` | fraction=0.001 | fraction=0.005 | fraction=0.01 | fraction=0.05 |
| `pitch_shift` | semitones=0.25 | semitones=0.5 | semitones=1.0 | semitones=2.0 |
| `sample_jitter` | events=2 | events=10 | events=50 | events=200 |
| `speed` | rate=1.01 | rate=0.99 | rate=1.05 | rate=0.95 |
| `time_shift` | shift_ms=1.0, shift_samples=None | shift_ms=10.0, shift_samples=None | shift_ms=100.0, shift_samples=None | shift_ms=250.0, shift_samples=None |
| `time_stretch` | rate=1.01 | rate=0.99 | rate=1.05 | rate=0.95 |
| `acoustic_channel` | rt60_seconds=0.2, snr_db=40.0, clock_offset_ppm=0.0 | rt60_seconds=0.5, snr_db=25.0, clock_offset_ppm=0.0 | rt60_seconds=1.0, snr_db=15.0, clock_offset_ppm=0.0 | rt60_seconds=1.5, snr_db=5.0, clock_offset_ppm=500.0 |
| `echo` | delay_ms=5.0, attenuation=0.1 | delay_ms=25.0, attenuation=0.25 | delay_ms=50.0, attenuation=0.5 | delay_ms=100.0, attenuation=0.5 |
| `reverb` | rt60_seconds=0.2 | rt60_seconds=0.5 | rt60_seconds=1.0 | rt60_seconds=1.5 |
<!-- /catalogue:attacks-severity -->

### Why these ranges

These are engineering benchmark choices, informed by common practice in the
audio watermarking literature. They are stated with their reasoning rather than
attributed to any particular publication.

- **SNR 40 → 5 dB.** 40 dB is at the edge of audibility on speech; 25 dB is
  clearly audible but undamaging; 15 dB is a poor channel; 5 dB is noise
  comparable to the signal, included to locate where a method finally fails.
- **Bitrates.** The ladder spans transparent (256/192 kbps) to the point where
  coding artefacts are plainly audible (64/32 kbps), which is the range in
  which distributed audio actually exists. Where an encoder cannot use a
  bitrate at the source rate, its levels and sweep start at the highest one
  it honours: MP3 at 16–24 kHz (MPEG-2) is limited to 160 kbps and below
  16 kHz to 64 kbps, which LAME would otherwise clamp silently while the row
  recorded the requested value; libvorbis rejects bitrates outside a
  rate-dependent range (for 16 kHz mono, nothing above 96 kbps).
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


## Attack reference

Generated from the `card` attribute of each class by `python -m taf.catalogue_docs`; edit the card, not this section. The same texts, in English and Polish, appear in the research UI.

<!-- catalogue:attacks-details -->
### Additive noise

#### White Gaussian noise { #awgn }

`awgn` · Additive noise · stochastic (seeded) · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/noise.py)

*Adds white Gaussian noise to model a noisy recording or transmission channel.*

Measures signal power and scales random noise to the requested `snr_db`. Lower SNR means stronger noise, while seed reproduces the random sequence. Noise perturbs sample values and transform coefficients throughout the recording; the resulting bit errors reveal the detector’s tolerance to distributed disturbance.

| Parameter | Default | Role |
| --- | --- | --- |
| `snr_db` | `20.0` | swept in robustness curves |
| `seed` | `0` | random seed; replaced per trial in experiments |
| `prevent_clipping` | `False` |  |

**Severity at 16 kHz:** mild `snr_db=40.0` · moderate `snr_db=25.0` · strong `snr_db=15.0` · extreme `snr_db=5.0`

**Sweep ladder:** `snr_db` = 40, 35, 30, 25, 20, 15, 10, 5, 0 (dB SNR)

#### Impulse noise { #impulse-noise }

`impulse_noise` · Additive noise · stochastic (seeded) · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/noise.py)

*Injects sparse strong impulses, resembling clicks or isolated transmission errors.*

Randomly selects a fraction of samples using density and sets impulse power from `snr_db`. At fixed total noise power, fewer affected samples mean stronger individual impulses. seed reproduces their positions. This distinguishes vulnerability to local damage from sensitivity to continuous background noise.

| Parameter | Default | Role |
| --- | --- | --- |
| `snr_db` | `20.0` | swept in robustness curves |
| `density` | `0.001` |  |
| `seed` | `0` | random seed; replaced per trial in experiments |
| `prevent_clipping` | `False` |  |

**Severity at 16 kHz:** mild `snr_db=40.0`, `density=0.001` · moderate `snr_db=25.0`, `density=0.005` · strong `snr_db=15.0`, `density=0.01` · extreme `snr_db=5.0`, `density=0.05`

**Sweep ladder:** `snr_db` = 40, 30, 20, 15, 10, 5 (dB SNR)

#### Pink noise { #pink-noise }

`pink_noise` · Additive noise · stochastic (seeded) · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/noise.py)

*Adds pink noise, concentrating more disturbance at low frequencies than white noise.*

Generates noise with an approximately 1/f power spectrum and scales it to the requested SNR. `snr_db` controls total power and seed controls reproducibility. At equal SNR, its frequency distribution differs from AWGN, so comparing them helps identify whether the payload depends on low-frequency content.

| Parameter | Default | Role |
| --- | --- | --- |
| `snr_db` | `20.0` | swept in robustness curves |
| `seed` | `0` | random seed; replaced per trial in experiments |
| `prevent_clipping` | `False` |  |

**Severity at 16 kHz:** mild `snr_db=40.0` · moderate `snr_db=25.0` · strong `snr_db=15.0` · extreme `snr_db=5.0`

**Sweep ladder:** `snr_db` = 40, 35, 30, 25, 20, 15, 10, 5, 0 (dB SNR)

### Lossy codecs

#### Lossy codec round trip { #codec }

`codec` · Lossy codecs · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/codec.py)

*Tests whether the message survives real lossy encoding and decoding (MP3, AAC, Opus or Vorbis).*

Runs the codec through FFmpeg and returns decoded PCM. `bitrate_kbps` sets the target bitrate; a lower rate usually discards more information. Optional delay alignment and length restoration separate codec damage from timing changes. Perceptual compression can remove quiet embedded components; results depend on the encoder and FFmpeg build.

| Parameter | Default | Role |
| --- | --- | --- |
| `codec` | `'mp3'` |  |
| `bitrate_kbps` | `128` |  |
| `align_delay` | `True` |  |
| `restore_length` | `True` |  |

Shortcuts: `mp3`, `aac`, `opus`, `vorbis` select the codec.

**Severity at 16 kHz:** mild `bitrate_kbps=160` · moderate `bitrate_kbps=128` · strong `bitrate_kbps=96` · extreme `bitrate_kbps=64`

### Filtering

#### Band-pass filter { #band-pass }

`band_pass` · Filtering · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/filtering.py)

*Keeps only a selected frequency band, modelling a bandwidth-limited channel.*

Combines lower and upper cutoffs in a Butterworth band-pass filter. Narrowing the passband removes more carrier information, as in telephone-like transmission. Filter order controls transition steepness and `zero_phase` controls phase handling. Payload stored outside the retained band is especially exposed.

| Parameter | Default | Role |
| --- | --- | --- |
| `low_hz` | `300.0` |  |
| `high_hz` | `3400.0` |  |
| `order` | `6` |  |
| `zero_phase` | `True` |  |

**Severity at 16 kHz:** mild `low_hz=20.0`, `high_hz=7200.0` · moderate `low_hz=100.0`, `high_hz=4000.0` · strong `low_hz=300.0`, `high_hz=2400.0` · extreme `low_hz=1000.0`, `high_hz=1200.0`

#### High-pass filter { #high-pass }

`high_pass` · Filtering · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/filtering.py)

*Attenuates low frequencies, as in rumble removal or an AC-coupled recording path.*

Applies a Butterworth high-pass filter with a chosen cutoff and order. Raising the cutoff removes more low-band content and can damage amplitude-relation or low-frequency transform embedding. `zero_phase` controls forward-backward versus causal filtering, separating band attenuation from phase-delay effects.

| Parameter | Default | Role |
| --- | --- | --- |
| `cutoff_hz` | `300.0` | swept in robustness curves |
| `order` | `6` |  |
| `zero_phase` | `True` |  |

**Severity at 16 kHz:** mild `cutoff_hz=20.0` · moderate `cutoff_hz=100.0` · strong `cutoff_hz=300.0` · extreme `cutoff_hz=1000.0`

**Sweep ladder:** `cutoff_hz` = 50, 100, 200, 300, 500, 800, 1000 (Hz)

#### Low-pass filter { #low-pass }

`low_pass` · Filtering · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/filtering.py)

*Attenuates high frequencies to test whether the watermark depends on the upper band.*

Uses a Butterworth low-pass filter specified by cutoff and order. The default forward-backward mode avoids group delay but applies filtering twice; causal mode also changes phase. Lower cutoff removes more bandwidth. Lost upper-band payload components cannot be recovered merely by restoring the original sample rate.

| Parameter | Default | Role |
| --- | --- | --- |
| `cutoff_hz` | `4000.0` | swept in robustness curves |
| `order` | `6` |  |
| `zero_phase` | `True` |  |

**Severity at 16 kHz:** mild `cutoff_hz=7200.0` · moderate `cutoff_hz=4000.0` · strong `cutoff_hz=2400.0` · extreme `cutoff_hz=1200.0`

**Sweep ladder:** `cutoff_hz` = 7200, 6000, 4800, 4000, 3200, 2400, 1600 (Hz)

#### Notch filter { #notch }

`notch` · Filtering · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/filtering.py)

*Suppresses a narrow frequency region, such as hum or a tonal interferer.*

Uses a second-order IIR notch around a centre frequency; its quality factor determines bandwidth. Optional `depth_db` allows partial attenuation instead of a full notch. It probes whether a mark relies on a narrow carrier region rather than broadly distributed energy.

| Parameter | Default | Role |
| --- | --- | --- |
| `center_hz` | `1000.0` |  |
| `quality` | `30.0` |  |
| `depth_db` | `None` |  |

**Severity at 16 kHz:** mild `center_hz=2000.0`, `quality=60.0` · moderate `center_hz=2000.0`, `quality=30.0` · strong `center_hz=2000.0`, `quality=10.0` · extreme `center_hz=2000.0`, `quality=3.0`

#### Moving-average smoothing { #smoothing }

`smoothing` · Filtering · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/filtering.py)

*Smooths samples by replacing them with local moving averages.*

Convolves the waveform with a uniform boxcar window. A longer window suppresses fast variations more strongly and produces spectral nulls related to the window length. This models simple smoothing or naive denoising and can erase fine sample-level payload changes.

| Parameter | Default | Role |
| --- | --- | --- |
| `window_length` | `5` | swept in robustness curves |
| `zero_phase` | `True` |  |

**Severity at 16 kHz:** mild `window_length=3` · moderate `window_length=5` · strong `window_length=9` · extreme `window_length=17`

**Sweep ladder:** `window_length` = 3, 5, 7, 11, 15 (samples)

### Resampling & clock

#### Clock drift { #clock-drift }

`clock_drift` · Resampling & clock · changes length or sample rate · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/resampling.py)

*Simulates a small mismatch between playback and recording clocks.*

Resamples by a factor derived from `offset_ppm` while leaving the nominal sample rate unchanged. Even a small mismatch accumulates into positional drift over a long recording. The attack tests whether a decoder can track a gradually changing sample grid, not simply a constant initial offset.

| Parameter | Default | Role |
| --- | --- | --- |
| `offset_ppm` | `100.0` | swept in robustness curves |

**Severity at 16 kHz:** mild `offset_ppm=10.0` · moderate `offset_ppm=50.0` · strong `offset_ppm=100.0` · extreme `offset_ppm=500.0`

**Sweep ladder:** `offset_ppm` = 10, 50, 100, 500, 1000 (ppm)

#### Resampling round trip { #resample }

`resample` · Resampling & clock · changes length or sample rate · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/resampling.py)

*Converts to an intermediate sample rate and back to test rate-conversion damage.*

Uses polyphase FIR resampling with anti-alias filtering. A lower `intermediate_hz` limits the recoverable bandwidth and changes the sample grid. Optional `restore_length` trims or pads rounding differences. Returning to the original rate lets the decoder use its expected rate, but does not restore discarded frequencies or original sample values.

| Parameter | Default | Role |
| --- | --- | --- |
| `intermediate_hz` | `22050` | swept in robustness curves |
| `restore_length` | `True` |  |

**Severity at 16 kHz:** mild `intermediate_hz=22050` · moderate `intermediate_hz=8000` · strong `intermediate_hz=8000` · extreme `intermediate_hz=8000`

**Sweep ladder:** `intermediate_hz` = 22050, 8000 (Hz)

### Quantisation

#### Bit-depth reduction { #bit-depth }

`bit_depth` · Quantisation · stochastic (seeded) · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/quantization.py)

*Requantises audio to fewer PCM amplitude levels.*

Maps samples to a uniform grid with spacing 2 / 2^bits over nominal full scale. The mode selects a grid with or without an exact zero level; optional TPDF dither adds noise before quantisation. Lower bit depth removes finer sample detail and is particularly destructive to payloads stored in low sample bits.

| Parameter | Default | Role |
| --- | --- | --- |
| `bits` | `8` | swept in robustness curves |
| `dither` | `False` |  |
| `mode` | `'mid_tread'` |  |
| `seed` | `0` | random seed; replaced per trial in experiments |

**Severity at 16 kHz:** mild `bits=16` · moderate `bits=12` · strong `bits=10` · extreme `bits=8`

**Sweep ladder:** `bits` = 16, 12, 10, 8, 6, 4 (bits)

### Amplitude & dynamics

#### Clipping { #clipping }

`clipping` · Amplitude & dynamics · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/amplitude.py)

*Limits signal peaks, modelling saturation or overdriven audio.*

Hard-clips samples at a threshold expressed relative to full scale, the signal peak or an amplitude percentile. A lower threshold affects more samples and reshapes the waveform, introducing additional spectral components. This tests payloads stored in peak values, amplitude statistics and transform coefficients without deliberately changing the time axis.

| Parameter | Default | Role |
| --- | --- | --- |
| `threshold` | `0.9` | swept in robustness curves |
| `mode` | `'peak'` |  |

**Severity at 16 kHz:** mild `threshold=0.95`, `mode='peak'` · moderate `threshold=0.9`, `mode='peak'` · strong `threshold=0.8`, `mode='peak'` · extreme `threshold=0.7`, `mode='peak'`

**Sweep ladder:** `threshold` = 0.99, 0.9, 0.8, 0.6, 0.4, 0.2 (× peak)

#### Dynamic-range compression { #compression-dynamic }

`compression_dynamic` · Amplitude & dynamics · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/amplitude.py)

*Reduces the difference between loud and quiet samples, then restores the original peak.*

A memoryless compressor: sample magnitudes above threshold (linear, relative to full scale) are reduced by ratio, and make-up gain returns the peak to its original level. There is no envelope follower and no attack or release time, so two numbers define the attack completely. It changes amplitude relations while leaving the audio listenable, which stresses amplitude- and norm-relation methods.

| Parameter | Default | Role |
| --- | --- | --- |
| `threshold` | `0.3` |  |
| `ratio` | `4.0` | swept in robustness curves |

**Severity at 16 kHz:** mild `ratio=2.0` · moderate `ratio=4.0` · strong `ratio=8.0` · extreme `ratio=16.0`

**Sweep ladder:** `ratio` = 2.0, 4.0, 8.0, 16.0 (: 1)

#### Gain change { #gain }

`gain` · Amplitude & dynamics · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/amplitude.py)

*Changes playback level to test dependence on absolute amplitude.*

Multiplies samples by the gain corresponding to the requested decibel change. Optional clipping limits values that exceed full scale and adds a separate nonlinear distortion. Pure gain leaves relative amplitude relations intact, whereas fixed absolute detection thresholds or quantisation steps may no longer match.

| Parameter | Default | Role |
| --- | --- | --- |
| `gain_db` | `-6.0` | swept in robustness curves |
| `prevent_clipping` | `False` |  |

**Severity at 16 kHz:** mild `gain_db=-3.0` · moderate `gain_db=-6.0` · strong `gain_db=-12.0` · extreme `gain_db=6.0`

**Sweep ladder:** `gain_db` = 0, -6, -12, -18, -24, -30 (dB)

### Temporal & desynchronisation

#### Cropping { #crop }

`crop` · Temporal & desynchronisation · stochastic (seeded) · changes length or sample rate · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/temporal.py)

*Removes part of the recording to test payload loss and framing sensitivity.*

Deletes the configured fraction from the start, end, both ends or a seeded random location. The result is shorter, so some bits may disappear and length-derived frame boundaries may move. It tests partial-recording recovery; success depends on redundancy, synchronisation and where the method stored its message.

| Parameter | Default | Role |
| --- | --- | --- |
| `fraction` | `0.01` | swept in robustness curves |
| `position` | `'start'` |  |
| `seed` | `0` | random seed; replaced per trial in experiments |

**Severity at 16 kHz:** mild `fraction=0.001` · moderate `fraction=0.01` · strong `fraction=0.05` · extreme `fraction=0.1`

**Sweep ladder:** `fraction` = 0.01, 0.02, 0.05, 0.1, 0.2 (fraction)

#### Sample dropout { #dropout }

`dropout` · Temporal & desynchronisation · stochastic (seeded) · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/temporal.py)

*Replaces short audio regions with zeros, modelling missing packets or recording dropouts.*

Selects runs of samples and silences them while preserving total length. Parameters determine how much audio is lost and the run size; seed reproduces placement. Unlike cropping, positions after a dropout remain aligned, helping distinguish loss of payload content from loss of synchronisation.

| Parameter | Default | Role |
| --- | --- | --- |
| `fraction` | `0.01` | swept in robustness curves |
| `run_length` | `20` |  |
| `seed` | `0` | random seed; replaced per trial in experiments |

**Severity at 16 kHz:** mild `fraction=0.001` · moderate `fraction=0.005` · strong `fraction=0.01` · extreme `fraction=0.05`

**Sweep ladder:** `fraction` = 0.001, 0.005, 0.01, 0.02, 0.05, 0.1 (fraction)

#### Pitch shift { #pitch-shift }

`pitch_shift` · Temporal & desynchronisation · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/temporal.py)

*Raises or lowers pitch while keeping approximately the same duration.*

Applies a pitch-shifting operation parameterised in semitones. The resynthesis changes frequency content and phase relationships even though the overall duration is maintained. It tests whether embedding relies on particular spectral positions or waveform detail, rather than simply on clip length.

| Parameter | Default | Role |
| --- | --- | --- |
| `semitones` | `0.5` | swept in robustness curves |

**Severity at 16 kHz:** mild `semitones=0.25` · moderate `semitones=0.5` · strong `semitones=1.0` · extreme `semitones=2.0`

**Sweep ladder:** `semitones` = 0.1, 0.25, 0.5, 1.0, 2.0 (semitones)

#### Sample insertion and deletion { #sample-jitter }

`sample_jitter` · Temporal & desynchronisation · stochastic (seeded) · changes length or sample rate · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/temporal.py)

*Inserts or deletes short sample runs at scattered positions.*

Deletion removes samples; insertion repeats preceding samples to model a simple concealment operation. Seeded positions make the attack reproducible. Each edit shifts subsequent boundaries, creating local and cumulative desynchronisation that a single global offset correction cannot necessarily repair.

| Parameter | Default | Role |
| --- | --- | --- |
| `events` | `10` | swept in robustness curves |
| `run_length` | `8` |  |
| `operation` | `'delete'` |  |
| `seed` | `0` | random seed; replaced per trial in experiments |

**Severity at 16 kHz:** mild `events=2` · moderate `events=10` · strong `events=50` · extreme `events=200`

**Sweep ladder:** `events` = 1, 5, 10, 20, 50 (events)

#### Speed change { #speed }

`speed` · Temporal & desynchronisation · changes length or sample rate · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/temporal.py)

*Changes playback speed, affecting both duration and pitch.*

Resamples the waveform to model faster or slower playback while keeping the nominal output rate. Faster playback shortens the clip and raises pitch; slower playback does the reverse. It probes timing and frequency dependence together, unlike time stretching, which aims to preserve pitch.

| Parameter | Default | Role |
| --- | --- | --- |
| `rate` | `1.01` | swept in robustness curves |

**Severity at 16 kHz:** mild `rate=1.01` · moderate `rate=0.99` · strong `rate=1.05` · extreme `rate=0.95`

**Sweep ladder:** `rate` = 1.005, 1.01, 1.02, 1.05, 1.1 (×)

#### Time shift { #time-shift }

`time_shift` · Temporal & desynchronisation · changes length or sample rate · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/temporal.py)

*Moves the recording in time to test decoder synchronisation.*

Uses `shift_samples` or `shift_ms` to delay or advance audio. Pad mode fills exposed positions with zeros while keeping length; circular mode wraps samples around. A small shift can move every decoding frame away from its embedded position even when much of the waveform itself is unchanged.

| Parameter | Default | Role |
| --- | --- | --- |
| `shift_samples` | `None` |  |
| `shift_ms` | `10.0` |  |
| `mode` | `'pad'` |  |

**Severity at 16 kHz:** mild `shift_ms=1.0`, `shift_samples=None` · moderate `shift_ms=10.0`, `shift_samples=None` · strong `shift_ms=100.0`, `shift_samples=None` · extreme `shift_ms=250.0`, `shift_samples=None`

#### Time stretch { #time-stretch }

`time_stretch` · Temporal & desynchronisation · changes length or sample rate · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/temporal.py)

*Changes duration while approximately preserving pitch.*

Uses a phase vocoder to resynthesise audio at a different tempo. This changes both the temporal grid and waveform phases, rather than merely moving existing samples. The rate controls the duration change. It can damage phase-based and frame-based payloads even when the tempo difference sounds small.

| Parameter | Default | Role |
| --- | --- | --- |
| `rate` | `1.01` | swept in robustness curves |

**Severity at 16 kHz:** mild `rate=1.01` · moderate `rate=0.99` · strong `rate=1.05` · extreme `rate=0.95`

**Sweep ladder:** `rate` = 1.005, 1.01, 1.02, 1.05, 1.1 (×)

#### Zero padding { #zero-padding }

`zero_padding` · Temporal & desynchronisation · changes length or sample rate · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/temporal.py)

*Adds silence at the start, the end or both, without changing the retained samples.*

Inserts zero-valued samples amounting to fraction of the signal length at position start, end or both. Leading silence moves every embedded position, so a decoder that counts samples from the start of the file reads each frame at the wrong offset; trailing silence changes only the total length, which matters to methods that derive their framing from it. Leading silence is routine when a clip is cut from a longer recording or a container adds priming samples.

| Parameter | Default | Role |
| --- | --- | --- |
| `fraction` | `0.1` |  |
| `position` | `'start'` |  |

### Acoustic channel

#### Acoustic channel { #acoustic-channel }

`acoustic_channel` · Acoustic channel · stochastic (seeded) · changes length or sample rate · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/acoustic.py)

*Combines several effects to approximate a loudspeaker-to-microphone channel.*

Chains acoustic and transmission effects such as bandwidth limitation, reverberation, noise and clock mismatch according to its parameters. Damage accumulates across stages, so surviving each effect separately does not establish survival of the combination. It is a reproducible synthetic channel and does not replace validation on real recording hardware.

| Parameter | Default | Role |
| --- | --- | --- |
| `band_low_hz` | `100.0` |  |
| `band_high_hz` | `7000.0` |  |
| `rt60_seconds` | `0.3` |  |
| `snr_db` | `25.0` |  |
| `gain_db` | `-3.0` |  |
| `clock_offset_ppm` | `0.0` |  |
| `seed` | `0` | random seed; replaced per trial in experiments |

**Severity at 16 kHz:** mild `rt60_seconds=0.2`, `snr_db=40.0`, `clock_offset_ppm=0.0` · moderate `rt60_seconds=0.5`, `snr_db=25.0`, `clock_offset_ppm=0.0` · strong `rt60_seconds=1.0`, `snr_db=15.0`, `clock_offset_ppm=0.0` · extreme `rt60_seconds=1.5`, `snr_db=5.0`, `clock_offset_ppm=500.0`

#### Echo { #echo }

`echo` · Acoustic channel · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/acoustic.py)

*Adds one delayed, attenuated copy of the signal, modelling a single acoustic reflection.*

Computes y[n] = x[n] + attenuation · x[n − D], with the delay D given by `delay_ms` so it means the same at any sampling rate. Below about 10 ms the echo colours the sound; above about 50 ms it is heard as a repetition. An echo-hiding decoder searches the cepstrum for a peak at its own delay, so an echo at a different delay inserts a competing peak: the attack is targeted at echo methods and should be reported as such. `prevent_clipping` limits the result to full scale.

| Parameter | Default | Role |
| --- | --- | --- |
| `delay_ms` | `25.0` |  |
| `attenuation` | `0.25` | swept in robustness curves |
| `prevent_clipping` | `True` |  |

**Severity at 16 kHz:** mild `delay_ms=5.0`, `attenuation=0.1` · moderate `delay_ms=25.0`, `attenuation=0.25` · strong `delay_ms=50.0`, `attenuation=0.5` · extreme `delay_ms=100.0`, `attenuation=0.5`

**Sweep ladder:** `attenuation` = 0.1, 0.25, 0.5, 0.75 (gain)

#### Reverberation { #reverb }

`reverb` · Acoustic channel · stochastic (seeded) · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/attacks/acoustic.py)

*Simulates reverberation by convolving audio with a generated room impulse response.*

Combines direct sound with a decaying reflection response controlled by reverberation and mixing parameters. The random response is reproducible with a seed. Reverberation smears energy over time and changes phase, stressing echo detectors and local frame statistics. This is a simulated room, not a measured playback-and-recording experiment.

| Parameter | Default | Role |
| --- | --- | --- |
| `rt60_seconds` | `0.5` | swept in robustness curves |
| `mix` | `1.0` |  |
| `seed` | `0` | random seed; replaced per trial in experiments |
| `trim_to_input` | `True` |  |

**Severity at 16 kHz:** mild `rt60_seconds=0.2` · moderate `rt60_seconds=0.5` · strong `rt60_seconds=1.0` · extreme `rt60_seconds=1.5`

**Sweep ladder:** `rt60_seconds` = 0.1, 0.3, 0.5, 0.8, 1.2 (s RT60)
<!-- /catalogue:attacks-details -->

See the [historical audit](attack-history.md) for changes to legacy attacks.
