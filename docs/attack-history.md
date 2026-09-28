# Attack implementation history

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
