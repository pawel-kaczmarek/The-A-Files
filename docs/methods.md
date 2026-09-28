# Embedding methods

The factory registers **27 methods**. The descriptions below state what the packaged code does.
References identify the underlying technique or related study; they do not certify an exact reproduction
of the paper’s implementation, training procedure or reported robustness. The review [1](references.md#ref-1)
provides background for the classical families, rather than a primary derivation of every local variant.

All methods accept a waveform and binary payload and expose blind extraction with the payload length.
Use registry identifiers in experiment configurations. Compare capacity, BER, audio distortion and
detectability separately under the [experimental protocol](experiments.md).

| Method / registry identifier | Mechanism and implementation scope | Reference |
| --- | --- | --- |
| **LSB**<br>`LSB_METHOD` | Replaces one least-significant bit of each float32 sample representation; conversion to integer PCM can erase the payload. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/LsbMethod.py). | [1](references.md#ref-1) |
| **Echo hiding**<br>`ECHO_METHOD` | Encodes bits using two echo delays; extraction compares cepstral evidence in each frame. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/EchoMethod.py). | [1](references.md#ref-1) |
| **Phase coding**<br>`PHASE_CODING_METHOD` | Stores bits in the first FFT block’s phase and preserves phase differences between successive blocks. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/PhaseCodingMethod.py). | [1](references.md#ref-1) |
| **Improved phase coding**<br>`IMPROVED_PHASE_CODING_METHOD` | Distributes phase-coded bits across FFT blocks while retaining the magnitude spectrum. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/ImprovedPhaseCodingMethod.py). | [19](references.md#ref-19) |
| **DCT-Delta-LSB**<br>`DCT_DELTA_LSB_METHOD` | Carries a bit in the relative norms of two DCT coefficient subsets; the local implementation uses a norm margin. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/DctDeltaLsbMethod.py). | [1](references.md#ref-1) |
| **DWT-LSB**<br>`DWT_LSB_METHOD` | Embeds parity in quantised wavelet detail coefficients using a step derived from band RMS. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/DwtLsbMethod.py). | [1](references.md#ref-1) |
| **DCT-b1**<br>`DCT_B1_METHOD` | Modifies first-band DCT coefficients with masking and energy compensation. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/DctB1Method.py). | [2](references.md#ref-2) |
| **Patchwork-ML**<br>`PATCHWORK_MULTILAYER_METHOD` | Changes the relative mean magnitudes of paired DCT groups; this implementation uses the first layer only. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/PatchworkMultilayerMethod.py). | [3](references.md#ref-3) |
| **Norm-space**<br>`NORM_SPACE_METHOD` | Encodes bits through the relative norms of interleaved DCT coefficients of a wavelet approximation. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/NormSpaceMethod.py). | [4](references.md#ref-4) |
| **FSVC**<br>`FSVC_METHOD` | Adjusts the singular-value ratio of selected DCT bands in paired frame halves. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/FsvcMethod.py). | [5](references.md#ref-5) |
| **DSSS**<br>`DSSS_METHOD` | Adds a keyed bipolar sequence per bit, scaled by frame RMS; decoding uses correlation. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/DsssMethod.py). | [6](references.md#ref-6) |
| **Blind SVD**<br>`BLIND_SVD_METHOD` | Selects a high-entropy DCT sub-band and quantises its largest singular value; a local adaptation of the cited scheme. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/BlindSvdMethod.py). | [20](references.md#ref-20) |
| **Prime-factor interpolation**<br>`PRIME_FACTOR_INTERPOLATE` | Places payload offsets at interpolated samples; retained neighbours determine variable bit capacity. Cover reversibility is not established by the TAF interface. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/PrimeFactorInterpolatedMethod.py). | [21](references.md#ref-21) |
| **LWT**<br>`LWT_METHOD` | Forces signs of eight-coefficient Haar detail blocks. The implementation uses PyWavelets Haar decomposition, not an explicit lifting pipeline. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/LwtMethod.py). | [22](references.md#ref-22) |
| **FBS-LSB**<br>`FBSMethod` | Separates foreground and background, then stores two or one float32 representation bits per selected sample. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/ForegroundBackgroundSegmentationMethod.py). | [23](references.md#ref-23) |
| **FGAS**<br>`FGAS_METHOD` | Optimises an audio perturbation against a fixed, seeded CNN decoder; requires the TensorFlow extra. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/FgasMethod.py). | [24](references.md#ref-24) |
| **AAC-STC**<br>`AAC_STC_METHOD` | Uses codec residuals to assign ±1 PCM costs and syndrome-trellis coding to minimise embedding cost; Vorbis is a fallback for AAC. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/AacStcMethod.py). | [25](references.md#ref-25) |
| **Wireless DWT-LSB**<br>`WIRELESS_DWT_LSB_METHOD` | Embeds supplied bits in low-frequency wavelet coefficients; adapts the paper’s carrier domain without its wireless transmission simulation. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/WirelessDwtLsbMethod.py). | [27](references.md#ref-27) |
| **LE-GA**<br>`LEARNABLE_EMBEDDING_GA_METHOD` | Uses deterministic waveform expansion, a uniform mask and correlation extraction in place of learned modules; an untrained approximation. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/LearnableEmbeddingGaMethod.py). | [28](references.md#ref-28) |
| **QIM / ST-DM**<br>`QIM_METHOD` | Quantises a keyed frame projection with bit-dependent dither; the orthogonal component determines the local step. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/QimMethod.py). | [29](references.md#ref-29) |
| **ISS**<br>`IMPROVED_SPREAD_SPECTRUM_METHOD` | Compensates the host projection before adding the spread bit, reducing host interference at the correlation detector. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/ImprovedSpreadSpectrumMethod.py). | [30](references.md#ref-30) |
| **Backward–forward echo**<br>`BACKWARD_FORWARD_ECHO_METHOD` | Uses paired backward and forward echo kernels; extraction compares their cepstral contributions. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/BackwardForwardEchoMethod.py). | [32](references.md#ref-32) |
| **Time-spread echo**<br>`TIME_SPREAD_ECHO_METHOD` | Spreads an echo with a keyed pseudo-noise kernel and recovers the delay through cepstral correlation. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/TimeSpreadEchoMethod.py). | [33](references.md#ref-33) |
| **Histogram**<br>`HISTOGRAM_METHOD` | Encodes bits in population relations among three adjacent amplitude bins, reducing dependence on sample order. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/HistogramMethod.py). | [34](references.md#ref-34) |
| **LFAM**<br>`LOW_FREQUENCY_AMPLITUDE_METHOD` | Modifies amplitude relations among three consecutive low-frequency subsegments. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/LowFrequencyAmplitudeMethod.py). | [35](references.md#ref-35) |
| **AudioSeal**<br>`AUDIOSEAL_METHOD` | Wraps released pretrained neural models; the TAF adapter divides longer payloads into 16-bit chunks. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/AudioSealMethod.py). | [36](references.md#ref-36) |
| **WavMark**<br>`WAVMARK_METHOD` | Wraps pretrained watermarking with synchronisation; each one-second window carries 16 user bits and 16 synchronisation bits. [Code](https://github.com/pawelkaczmarek12/the-a-files/blob/master/src/taf/methods/WavMarkMethod.py). | [37](references.md#ref-37) |

## Interpretation

Steganographic security concerns detecting the *presence* of a payload; watermark robustness concerns
recovering it after processing. High SNR alone establishes neither. Paper-reported performance must retain
its corpus, payload rate, attack severity and detector assumptions when compared with local results.

The LSB and FBS variants work on floating-point representations. A successful in-memory round trip
does not imply survival after PCM export. Patchwork-ML, LWT and LE-GA have explicit implementation
restrictions in the table; especially LE-GA must not be reported as a reproduced trained neural system.

AudioSeal and WavMark require `the-a-files[neural]` and pretrained weights. FGAS requires
`the-a-files[ai]`. See [installation](installation.md) and [provenance](experiments.md#provenance).
