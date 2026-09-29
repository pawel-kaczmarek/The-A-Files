# Embedding methods

The descriptions below state what the packaged code does.
References identify the underlying technique or related study; they do not certify an exact reproduction
of the paper’s implementation, training procedure or reported robustness. The review [1](references.md#ref-1)
provides background for the classical families, rather than a primary derivation of every local variant.

All methods accept a waveform and binary payload and expose blind extraction with the payload length.
Use registry identifiers in experiment configurations. Compare capacity, BER, audio distortion and
detectability separately under the [experimental protocol](experiments.md).

<!-- catalogue:methods-table -->
**30 methods** are registered. Select a name for its mechanism, parameters and limits.

| Method / registry identifier | What it does | Family · purpose | Reference |
| --- | --- | --- | --- |
| [**FBS-LSB**](#fbsmethod)<br>`FBSMethod` | Adapts LSB capacity to foreground and background regions of the recording. | Sample-domain LSB · steganography | [23](references.md#ref-23) |
| [**LSB**](#lsb-method)<br>`LSB_METHOD` | Hides one message bit in each audio sample with a very small numerical change. | Sample-domain LSB · steganography | [1](references.md#ref-1) |
| [**Prime-factor interpolation**](#prime-factor-interpolate)<br>`PRIME_FACTOR_INTERPOLATE` | Stores a variable number of bits as small offsets from interpolated sample values. | Sample-domain LSB · steganography | [21](references.md#ref-21) |
| [**Blind SVD**](#blind-svd-method)<br>`BLIND_SVD_METHOD` | Embeds one bit per frame in the largest singular value of an entropy-selected DCT sub-band. | Transform domain · watermarking | [20](references.md#ref-20) |
| [**DCT-b1**](#dct-b1-method)<br>`DCT_B1_METHOD` | Embeds a watermark in a selected DCT band using masking and energy compensation. | Transform domain · watermarking | [2](references.md#ref-2) |
| [**DCT-Delta-LSB**](#dct-delta-lsb-method)<br>`DCT_DELTA_LSB_METHOD` | Encodes bits in the relative norms of two groups of cosine-transform coefficients. | Transform domain · steganography | [1](references.md#ref-1) |
| [**DWT-LSB**](#dwt-lsb-method)<br>`DWT_LSB_METHOD` | Hides bits in the parity of quantised wavelet detail coefficients. | Transform domain · steganography | [1](references.md#ref-1) |
| [**EMD**](#emd-method)<br>`EMD_METHOD` | Encodes bits in extrema of a slow component obtained by empirical mode decomposition. | Transform domain · watermarking | [48](references.md#ref-48) |
| [**FSVC**](#fsvc-method)<br>`FSVC_METHOD` | Stores bits in the relative singular values of two frequency-domain frame halves. | Transform domain · watermarking | [5](references.md#ref-5) |
| [**LWT**](#lwt-method)<br>`LWT_METHOD` | Encodes bits as the sign of blocks of wavelet detail coefficients. | Transform domain · watermarking | [22](references.md#ref-22) |
| [**Norm-space**](#norm-space-method)<br>`NORM_SPACE_METHOD` | Represents each bit by which of two transform-domain vectors has the larger norm. | Transform domain · watermarking | [4](references.md#ref-4) |
| [**Sync-DWT-DCT**](#sync-dwt-dct-method)<br>`SYNC_DWT_DCT_METHOD` | Combines transform-domain data with synchronisation codes to locate marked blocks after timing changes. | Transform domain · watermarking | [47](references.md#ref-47) |
| [**Wireless DWT-LSB**](#wireless-dwt-lsb-method)<br>`WIRELESS_DWT_LSB_METHOD` | Places payload bits in the low-frequency approximation coefficients of a wavelet transform. | Transform domain · steganography | [27](references.md#ref-27) |
| [**DSSS**](#dsss-method)<br>`DSSS_METHOD` | Spreads each bit over a frame using a key-derived pseudo-noise sequence. | Spread spectrum · steganography | [6](references.md#ref-6) |
| [**ISS**](#improved-spread-spectrum-method)<br>`IMPROVED_SPREAD_SPECTRUM_METHOD` | Reduces host-signal interference before adding a spread-spectrum watermark. | Spread spectrum · watermarking | [30](references.md#ref-30) |
| [**Backward–forward echo**](#backward-forward-echo-method)<br>`BACKWARD_FORWARD_ECHO_METHOD` | Carries bits with paired backward and forward echoes. | Echo hiding · watermarking | [32](references.md#ref-32) |
| [**Echo hiding**](#echo-method)<br>`ECHO_METHOD` | Represents zero and one by echoes with different delays. | Echo hiding · steganography | [1](references.md#ref-1) |
| [**Time-spread echo**](#time-spread-echo-method)<br>`TIME_SPREAD_ECHO_METHOD` | Distributes the echo watermark over multiple delays using a keyed sequence. | Echo hiding · watermarking | [33](references.md#ref-33) |
| [**Improved phase coding**](#improved-phase-coding-method)<br>`IMPROVED_PHASE_CODING_METHOD` | Distributes phase-coded message portions across multiple Fourier blocks. | Phase coding · steganography | [19](references.md#ref-19) |
| [**Phase coding**](#phase-coding-method)<br>`PHASE_CODING_METHOD` | Stores the message in Fourier phases of the first audio block. | Phase coding · steganography | [1](references.md#ref-1) |
| [**QIM / ST-DM**](#qim-method)<br>`QIM_METHOD` | Encodes bits by choosing between two quantisation grids for a keyed frame projection. | Quantisation (QIM) · watermarking | [29](references.md#ref-29) |
| [**Histogram**](#histogram-method)<br>`HISTOGRAM_METHOD` | Stores bits in the distribution of sample amplitudes rather than their positions. | Statistical / patchwork · watermarking | [34](references.md#ref-34) |
| [**LFAM**](#low-frequency-amplitude-method)<br>`LOW_FREQUENCY_AMPLITUDE_METHOD` | Encodes bits through amplitude relations between three low-frequency sub-segments. | Statistical / patchwork · watermarking | [35](references.md#ref-35) |
| [**Patchwork-ML**](#patchwork-multilayer-method)<br>`PATCHWORK_MULTILAYER_METHOD` | Carries bits in statistical differences between paired DCT coefficient groups. | Statistical / patchwork · watermarking | [3](references.md#ref-3) |
| [**AAC-STC**](#aac-stc-method)<br>`AAC_STC_METHOD` | Chooses low-cost sample edits using perceptual codec residuals and syndrome-trellis coding. | Adaptive coding · steganography | [25](references.md#ref-25) |
| [**PEE**](#reversible-pee-method)<br>`REVERSIBLE_PEE_METHOD` | Hides data while allowing exact recovery of the original 16-bit PCM cover when no damage occurs. | Reversible (lossless) · steganography | [49](references.md#ref-49), [50](references.md#ref-50) |
| [**LE-GA**](#learnable-embedding-ga-method)<br>`LEARNABLE_EMBEDDING_GA_METHOD` | Provides an experimental embedding pipeline inspired by learnable watermarking and genetic optimisation. | Learned embedding · watermarking | [28](references.md#ref-28) |
| [**AudioSeal**](#audioseal-method)<br>`AUDIOSEAL_METHOD` | Uses pretrained AudioSeal networks to generate and detect a neural audio watermark. | Neural network · watermarking | [36](references.md#ref-36) |
| [**FGAS**](#fgas-method)<br>`FGAS_METHOD` | Optimises a small audio perturbation so a fixed neural decoder outputs the desired bits. | Neural network · steganography | [24](references.md#ref-24) |
| [**WavMark**](#wavmark-method)<br>`WAVMARK_METHOD` | Uses the pretrained WavMark neural model to embed and recover bit payloads in audio chunks. | Neural network · watermarking | [37](references.md#ref-37) |
<!-- /catalogue:methods-table -->

## Interpretation

Steganographic security concerns detecting the *presence* of a payload; watermark robustness concerns
recovering it after processing. High SNR alone establishes neither. Paper-reported performance must retain
its corpus, payload rate, attack severity and detector assumptions when compared with local results.

The LSB and FBS variants work on floating-point representations. A successful in-memory round trip
does not imply survival after PCM export. Patchwork-ML, LWT and LE-GA have explicit implementation
restrictions in the table; especially LE-GA must not be reported as a reproduced trained neural system.

Sync-DWT-DCT is the only classical method that searches the signal for a synchronisation code; frame-based
methods that read a fixed grid lose synchronisation when samples are removed or inserted at the start. It reads a message
longer than one block (32 bits by default) by numbering blocks from the first one found, so cropping the start
preserves only a message that fits in one block. EMD depends on a data-driven decomposition that processing
itself changes: low-pass filtering in particular alters which content forms the carrier, so the method is
markedly less robust to filtering than fixed-basis transforms. PEE is reversible, not robust: any change to the
marked samples, including requantisation, destroys both the payload and the ability to restore the cover, and
a cover that is not on the 16-bit grid is restored in its rounded form.

AudioSeal and WavMark require `the-a-files[neural]` and pretrained weights. FGAS requires
`the-a-files[ai]`. See [installation](installation.md) and [provenance](experiments.md#provenance).

## Method reference

Generated from the `card` attribute of each class by `python -m taf.catalogue_docs`; edit the card, not this section. The same texts, in English and Polish, appear in the research UI.

<!-- catalogue:methods-details -->
### Sample-domain LSB

#### FBS-LSB { #fbsmethod }

`FBSMethod` · Sample-domain LSB · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/ForegroundBackgroundSegmentationMethod.py)

*Adapts LSB capacity to foreground and background regions of the recording.*

Separates foreground from background, shuffles sample positions with a shared seed, then stores two float-representation bits per foreground sample and one per background sample. Decoding repeats the segmentation and ordering. Both sample precision and stable segmentation matter; format conversion or processing can break extraction.

| Parameter | Default | Role |
| --- | --- | --- |
| `seed` | `42` | secret key |

**Reference:** Wang & Wang (2025) [[23](references.md#ref-23)]

#### LSB { #lsb-method }

`LSB_METHOD` · Sample-domain LSB · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/LsbMethod.py)

*Hides one message bit in each audio sample with a very small numerical change.*

Replaces the least significant bit of the sample’s 32-bit floating-point representation. Decoding reads those bits in order. This implementation uses float bits, not integer PCM bits: conversion to PCM, lossy compression or even small signal processing changes can erase the message.

No tunable parameters.

**Reference:** Alsabhany et al. (2020) [[1](references.md#ref-1)]

#### Prime-factor interpolation { #prime-factor-interpolate }

`PRIME_FACTOR_INTERPOLATE` · Sample-domain LSB · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/PrimeFactorInterpolatedMethod.py)

*Stores a variable number of bits as small offsets from interpolated sample values.*

Every second sample is replaced by an interpolation of its retained neighbours plus the payload value. Capacity follows the least prime factor of a log-scaled neighbour difference, capped by `max_bits_per_sample`. The decoder derives the same prediction and capacity from the neighbours. More bits mean larger offsets; changing the neighbours can corrupt decoding. Recovering the original cover is not part of the TAF interface, so reversibility is not established here.

| Parameter | Default | Role |
| --- | --- | --- |
| `max_bits_per_sample` | `4` |  |

**Reference:** Adhiyaksa et al. (2022) [[21](references.md#ref-21)]

### Transform domain

#### Blind SVD { #blind-svd-method }

`BLIND_SVD_METHOD` · Transform domain · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/BlindSvdMethod.py)

*Embeds one bit per frame in the largest singular value of an entropy-selected DCT sub-band.*

Chooses a low-frequency sub-band with maximum estimated entropy, reshapes it into a matrix and applies SVD. Quantisation parity of the largest singular value carries the bit. The step is derived from the remaining singular values and `quantization_coefficient`. No original cover is needed, but processing that changes sub-band selection can disrupt detection.

| Parameter | Default | Role |
| --- | --- | --- |
| `frame_size` | `1024` |  |
| `sub_band_count` | `4` |  |
| `quantization_coefficient` | `0.1` | embedding strength |

**Reference:** Dhar & Shimamura (2015) [[20](references.md#ref-20)]

#### DCT-b1 { #dct-b1-method }

`DCT_B1_METHOD` · Transform domain · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/DctB1Method.py)

*Embeds a watermark in a selected DCT band using masking and energy compensation.*

Processes audio in frames, changes coefficient groups to represent bits, and compensates energy changes while reconstructing the waveform. The detector evaluates the corresponding coefficient relations. Frame and group lengths control capacity and the embedding region; preserving frame alignment is essential after temporal attacks.

| Parameter | Default | Role |
| --- | --- | --- |
| `lt` | `23` |  |
| `lw` | `1486` |  |
| `lG1` | `24` |  |
| `lG2` | `6` |  |
| `key` | `20240521` | secret key |

**Reference:** Hu & Hsu (2015) [[2](references.md#ref-2)]

#### DCT-Delta-LSB { #dct-delta-lsb-method }

`DCT_DELTA_LSB_METHOD` · Transform domain · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/DctDeltaLsbMethod.py)

*Encodes bits in the relative norms of two groups of cosine-transform coefficients.*

Transforms each frame with DCT, splits coefficients into two vectors and makes one norm larger than the other according to the bit. Decoding compares the norms. Despite the historical LSB name, this implementation modifies norm relations. `delta_value` sets the separation relative to the frame level; larger separation increases distortion and the decision margin.

| Parameter | Default | Role |
| --- | --- | --- |
| `frame_length_in_ms` | `100` |  |
| `delta_value` | `0.05` | embedding strength |

**Reference:** Alsabhany et al. (2020) [[1](references.md#ref-1)]

#### DWT-LSB { #dwt-lsb-method }

`DWT_LSB_METHOD` · Transform domain · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/DwtLsbMethod.py)

*Hides bits in the parity of quantised wavelet detail coefficients.*

A two-level discrete wavelet transform separates the signal into approximation and detail bands. Selected detail coefficients are quantised to even or odd indices and decoded by parity. `step_scale` controls the step relative to detail-band RMS, making it follow signal level. Filtering and requantisation can still move coefficients across decision boundaries.

| Parameter | Default | Role |
| --- | --- | --- |
| `dwt_type` | `'bior5.5'` |  |
| `step_scale` | `0.5` | embedding strength |
| `spacing` | `10` |  |

**Reference:** Alsabhany et al. (2020) [[1](references.md#ref-1)]

#### EMD { #emd-method }

`EMD_METHOD` · Transform domain · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/EmdMethod.py)

*Encodes bits in extrema of a slow component obtained by empirical mode decomposition.*

Decomposes frames into oscillatory modes and quantises extrema of the remainder after the selected IMFs. It re-decomposes and corrects embedding in a closed loop; extrema vote on one bit per frame. This differs from marking only the last IMF in the paper. Filtering can change the decomposition itself, while repeated decomposition also increases computation cost. The paper's synchronisation code is not reproduced; Sync-DWT-DCT provides a sync-code scheme.

| Parameter | Default | Role |
| --- | --- | --- |
| `frame_length` | `1024` |  |
| `imf_count` | `4` |  |
| `step_scale` | `0.2` | embedding strength |
| `sift_iterations` | `4` |  |
| `edge_margin` | `64` |  |
| `max_refinements` | `6` |  |

**Reference:** Khaldi & Boudraa (2013) [[48](references.md#ref-48)]

#### FSVC { #fsvc-method }

`FSVC_METHOD` · Transform domain · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/FsvcMethod.py)

*Stores bits in the relative singular values of two frequency-domain frame halves.*

Splits each payload frame in half, applies DCT, selects coefficient bands and computes their singular values. Embedding enforces a bit-dependent ratio; decoding compares the values. In the current implementation the ratio is controlled by an internal alpha; the exposed delta parameter is stored but does not affect encoding. Temporal misalignment changes the compared halves.

| Parameter | Default | Role |
| --- | --- | --- |
| `delta` | `3.5793656` | embedding strength |

**Reference:** Zhao et al. (2021) [[5](references.md#ref-5)]

#### LWT { #lwt-method }

`LWT_METHOD` · Transform domain · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/LwtMethod.py)

*Encodes bits as the sign of blocks of wavelet detail coefficients.*

This implementation uses a two-level Haar wavelet decomposition and groups eight detail coefficients per bit. It forces a positive or negative block with a minimum magnitude set by threshold, then reconstructs the signal. Decoding reads the sign of the block mean. A larger threshold creates a stronger mark and more distortion; alignment and detail-band preservation matter. The transform is the PyWavelets Haar decomposition rather than an explicit lifting scheme.

| Parameter | Default | Role |
| --- | --- | --- |
| `threshold` | `0.05` | embedding strength |

**Reference:** Mushtaq et al. (2024) [[22](references.md#ref-22)]

#### Norm-space { #norm-space-method }

`NORM_SPACE_METHOD` · Transform domain · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/NormSpaceMethod.py)

*Represents each bit by which of two transform-domain vectors has the larger norm.*

Applies a Haar DWT and then DCT to the approximation band. Even and odd DCT coefficients form two vectors whose norms are separated by a relative delta. The decoder compares them without the original recording. Increasing delta strengthens the separation but changes audio more; cropping can break the segment grid.

| Parameter | Default | Role |
| --- | --- | --- |
| `delta` | `0.05` | embedding strength |

**Reference:** Saadi et al. (2019) [[4](references.md#ref-4)]

#### Sync-DWT-DCT { #sync-dwt-dct-method }

`SYNC_DWT_DCT_METHOD` · Transform domain · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/SyncDwtDctMethod.py)

*Combines transform-domain data with synchronisation codes to locate marked blocks after timing changes.*

Writes a Barker-based sync pattern through quantised sample-group means, followed by payload quantisation in the DCT of a DWT approximation band. The decoder scans offsets and validates candidate blocks. Relative quantisation steps follow signal level. Coefficient selection and step rules are implementation choices; successful resynchronisation still depends on the attack and remaining audio. The sync code has 16 bits (the 13-bit Barker code followed by a 3-bit one), one bit per `sync_group` samples; 16-sample groups replace the paper's 5. Because the decoder searches every offset, it finds the blocks again after shifting, padding or cropping.

| Parameter | Default | Role |
| --- | --- | --- |
| `segment_length` | `4096` |  |
| `sync_group` | `16` |  |
| `level` | `3` |  |
| `wavelet` | `'db4'` |  |
| `band_start` | `200.0` |  |
| `bits_per_block` | `32` |  |
| `step_scale` | `0.5` | embedding strength |
| `sync_scale` | `0.3` |  |
| `detection_threshold` | `0.2` |  |

**Reference:** Wang & Zhao (2006) [[47](references.md#ref-47)]

#### Wireless DWT-LSB { #wireless-dwt-lsb-method }

`WIRELESS_DWT_LSB_METHOD` · Transform domain · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/WirelessDwtLsbMethod.py)

*Places payload bits in the low-frequency approximation coefficients of a wavelet transform.*

Scales and rounds DWT approximation coefficients to integers, replaces up to `lsb_depth` low bits and reconstructs the waveform. Decoding repeats the transform and scaling. Wavelet type, level and `coefficient_scale` affect capacity and distortion. This adapts the paper’s carrier idea to direct bit payloads; the name does not imply guaranteed survival of a wireless channel.

| Parameter | Default | Role |
| --- | --- | --- |
| `dwt_type` | `'haar'` |  |
| `level` | `1` |  |
| `lsb_depth` | `8` |  |
| `coefficient_scale` | `22000` |  |

**Reference:** Hamdi et al. (2025) [[27](references.md#ref-27)]

### Spread spectrum

#### DSSS { #dsss-method }

`DSSS_METHOD` · Spread spectrum · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/DsssMethod.py)

*Spreads each bit over a frame using a key-derived pseudo-noise sequence.*

Adds a bipolar sequence with sign determined by the bit and amplitude proportional to local RMS and alpha. The receiver correlates the frame with the same sequence and reads the correlation sign. Spreading distributes the payload over many samples, but the cover itself interferes with detection. Larger alpha improves the detection margin at the cost of more added noise.

| Parameter | Default | Role |
| --- | --- | --- |
| `key` | `20240521` | secret key |
| `alpha` | `0.05` | embedding strength |
| `min_chip_length` | `1024` |  |

**Reference:** Nugraha (2011) [[6](references.md#ref-6)] · **Needs long input:** short files produce failed trials.

#### ISS { #improved-spread-spectrum-method }

`IMPROVED_SPREAD_SPECTRUM_METHOD` · Spread spectrum · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/ImprovedSpreadSpectrumMethod.py)

*Reduces host-signal interference before adding a spread-spectrum watermark.*

Projects each carrier frame onto a key-derived chip sequence and compensates its existing component before setting a bit-dependent projection. The decoder reads the correlation sign. This removes a source of ambiguity present in plain DSSS. strength controls the mark level; compensation itself also changes the carrier, and frame synchronisation is still required.

| Parameter | Default | Role |
| --- | --- | --- |
| `key` | `20240521` | secret key |
| `strength` | `0.05` | embedding strength |
| `removal` | `1.0` |  |
| `min_chip_length` | `256` |  |

**Reference:** Malvar & Florencio (2003) [[30](references.md#ref-30)]

### Echo hiding

#### Backward–forward echo { #backward-forward-echo-method }

`BACKWARD_FORWARD_ECHO_METHOD` · Echo hiding · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/BackwardForwardEchoMethod.py)

*Carries bits with paired backward and forward echoes.*

Uses echo kernels on both sides of a sample position with opposite signs. The detector compares the corresponding cepstral responses rather than relying on a single echo peak. alpha controls the added echo strength. The paired structure changes the detection statistic, but does not remove sensitivity to reverberation or lost frame alignment.

| Parameter | Default | Role |
| --- | --- | --- |
| `alpha` | `0.1` | embedding strength |
| `d0` | `150` |  |
| `d1` | `200` |  |
| `min_frame_length` | `2048` |  |

**Reference:** Kim & Choi (2003) [[32](references.md#ref-32)]

#### Echo hiding { #echo-method }

`ECHO_METHOD` · Echo hiding · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/EchoMethod.py)

*Represents zero and one by echoes with different delays.*

Splits the recording into payload frames and mixes in a delayed copy using alpha. The decoder detects the selected delay through cepstral analysis. Echo strength trades audibility against detection margin. Short frames, existing reverberation or temporal changes can make the two delay hypotheses difficult to distinguish.

| Parameter | Default | Role |
| --- | --- | --- |
| `alpha` | `0.2` | embedding strength |
| `d0` | `150` |  |
| `d1` | `200` |  |
| `min_frame_length` | `2048` |  |

**Reference:** Alsabhany et al. (2020) [[1](references.md#ref-1)] · **Needs long input:** short files produce failed trials.

#### Time-spread echo { #time-spread-echo-method }

`TIME_SPREAD_ECHO_METHOD` · Echo hiding · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/TimeSpreadEchoMethod.py)

*Distributes the echo watermark over multiple delays using a keyed sequence.*

A pseudo-noise kernel spreads echo energy in time instead of concentrating it at one delay. The decoder uses the corresponding sequence to detect the bit from the cepstral response. alpha sets strength and the shared seed reproduces the kernel. Spreading changes the audibility and detection trade-off; it still needs enough audio per bit and correct alignment.

| Parameter | Default | Role |
| --- | --- | --- |
| `key` | `20240521` | secret key |
| `alpha` | `0.1` | embedding strength |
| `pn_length` | `511` |  |
| `d0` | `200` |  |
| `d1` | `900` |  |
| `min_frame_length` | `4096` |  |

**Reference:** Ko et al. (2005) [[33](references.md#ref-33)]

### Phase coding

#### Improved phase coding { #improved-phase-coding-method }

`IMPROVED_PHASE_CODING_METHOD` · Phase coding · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/ImprovedPhaseCodingMethod.py)

*Distributes phase-coded message portions across multiple Fourier blocks.*

Divides the payload among segments and sets selected phase pairs to ±π/2 while retaining spectral magnitudes. The decoder reconstructs the segment layout from the message length and reads phase signs. Distribution avoids concentrating all bits in the first segment, but it is not error correction; altered length, phase distortion and trimming can still corrupt bits.

No tunable parameters.

**Reference:** Yang (2024) [[19](references.md#ref-19)]

#### Phase coding { #phase-coding-method }

`PHASE_CODING_METHOD` · Phase coding · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/PhaseCodingMethod.py)

*Stores the message in Fourier phases of the first audio block.*

Sets selected phases to +π/2 or −π/2 and mirrors them to preserve a real waveform. Subsequent blocks retain their original inter-block phase differences. Decoding reads the signs of the first block’s phases. Magnitudes are retained during embedding, but a shift or removal of the opening block can destroy access to the payload.

No tunable parameters.

**Reference:** Alsabhany et al. (2020) [[1](references.md#ref-1)]

### Quantisation (QIM)

#### QIM / ST-DM { #qim-method }

`QIM_METHOD` · Quantisation (QIM) · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/QimMethod.py)

*Encodes bits by choosing between two quantisation grids for a keyed frame projection.*

Projects a frame onto a key-derived direction and moves that projection to the nearest point on the bit’s dithered grid. The decoder chooses the closer grid. The step follows energy orthogonal to the projection, which embedding leaves unchanged, and is scaled by `step_scale`. Larger steps increase separation and distortion; timing changes disrupt the frame projections.

| Parameter | Default | Role |
| --- | --- | --- |
| `key` | `20240521` | secret key |
| `step_scale` | `0.1` | embedding strength |
| `min_frame_length` | `256` |  |

**Reference:** Chen & Wornell (2001) [[29](references.md#ref-29)]

### Statistical / patchwork

#### Histogram { #histogram-method }

`HISTOGRAM_METHOD` · Statistical / patchwork · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/HistogramMethod.py)

*Stores bits in the distribution of sample amplitudes rather than their positions.*

Forms an amplitude histogram and modifies population relations in groups of three neighbouring bins. The histogram range is relative to mean absolute amplitude; threshold controls the required relation. Because sample order is not used, the method targets timing and cropping robustness. Changes that reshape amplitude statistics, such as clipping or noise, can still destroy the relation.

| Parameter | Default | Role |
| --- | --- | --- |
| `amplitude_span` | `2.5` |  |
| `threshold` | `2.0` | embedding strength |
| `cutoff` | `2000.0` |  |
| `rounds` | `24` |  |
| `search_span` | `0.2` |  |
| `search_steps` | `41` |  |
| `min_samples_per_bin` | `32` |  |

**Reference:** Xiang & Huang (2007) [[34](references.md#ref-34)]

#### LFAM { #low-frequency-amplitude-method }

`LOW_FREQUENCY_AMPLITUDE_METHOD` · Statistical / patchwork · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/LowFrequencyAmplitudeMethod.py)

*Encodes bits through amplitude relations between three low-frequency sub-segments.*

Separates low-frequency content and scales consecutive sub-segments so the middle amplitude lies above or below the average of its neighbours. The decoder reads that relation without the original. margin controls the separation. Relative amplitudes tolerate common gain changes, but high-pass filtering and temporal misalignment can remove or rearrange the carrier.

| Parameter | Default | Role |
| --- | --- | --- |
| `cutoff` | `2000.0` |  |
| `margin` | `0.3` | embedding strength |
| `min_segment_length` | `768` |  |

**Reference:** Lie & Chang (2006) [[35](references.md#ref-35)]

#### Patchwork-ML { #patchwork-multilayer-method }

`PATCHWORK_MULTILAYER_METHOD` · Statistical / patchwork · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/PatchworkMultilayerMethod.py)

*Carries bits in statistical differences between paired DCT coefficient groups.*

Reorders coefficients in a selected band and adjusts the mean absolute values of paired segments so their ordering represents a bit. Extraction compares these means. Despite the multilayer name, the current code implements only the first layer. `min_segment_length` limits capacity; band removal and changes to the global transform grid can damage the mark.

| Parameter | Default | Role |
| --- | --- | --- |
| `min_segment_length` | `8` |  |

**Reference:** Natgunanathan et al. (2017) [[3](references.md#ref-3)]

### Adaptive coding

#### AAC-STC { #aac-stc-method }

`AAC_STC_METHOD` · Adaptive coding · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/AacStcMethod.py)

*Chooses low-cost sample edits using perceptual codec residuals and syndrome-trellis coding.*

Converts audio to 16-bit PCM and estimates embedding costs from the difference after an AAC round trip, with Vorbis as a fallback. A trellis search selects inexpensive ±1 changes whose LSB syndrome encodes the message. Decoding applies the same parity-check matrix. The codec guides where to embed; it does not make the resulting LSB payload inherently resistant to transcoding.

| Parameter | Default | Role |
| --- | --- | --- |
| `bitrate` | `400000` |  |
| `constraint_height` | `7` |  |
| `hhat_seed` | `0` | secret key |

**Reference:** Luo et al. (2017) [[25](references.md#ref-25)]

### Reversible (lossless)

#### PEE { #reversible-pee-method }

`REVERSIBLE_PEE_METHOD` · Reversible (lossless) · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/ReversiblePeeMethod.py)

*Hides data while allowing exact recovery of the original 16-bit PCM cover when no damage occurs.*

Predicts each sample from its two predecessors and expands small prediction errors as 2e + bit; larger errors are shifted. A location map handles overflow risks and decoding reverses the mappings. threshold controls eligible errors and capacity. Reversibility refers to the cover rounded onto the 16-bit grid; processing or another bit-depth conversion can destroy both payload and restoration. recover_cover() returns the original cover exactly from an undamaged stego signal. The method is fragile by design.

| Parameter | Default | Role |
| --- | --- | --- |
| `threshold` | `8` | embedding strength |

**Reference:** Thodi & Rodriguez (2007) [[49](references.md#ref-49)], Nishimura (2011) [[50](references.md#ref-50)]

### Learned embedding

#### LE-GA { #learnable-embedding-ga-method }

`LEARNABLE_EMBEDDING_GA_METHOD` · Learned embedding · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/LearnableEmbeddingGaMethod.py)

*Provides an experimental embedding pipeline inspired by learnable watermarking and genetic optimisation.*

Maps bits to an upscaled waveform, applies a mask and adds a scaled residual; extraction uses correlation-style pooling. Without the paper’s trained weights, this implementation uses deterministic minimal operators and an all-ones mask. `embedding_strength` sets the residual level. Treat it as an implementation-specific baseline, not a reproduction of the trained network’s reported performance.

| Parameter | Default | Role |
| --- | --- | --- |
| `scaling_factor` | `0.9` |  |
| `embedding_strength` | `0.005` | embedding strength |
| `population_size` | `10` |  |
| `mutation_rate` | `0.1` |  |
| `mutation_std` | `0.05` |  |
| `seed` | `42` | secret key |
| `min_samples_per_bit` | `16` |  |

**Reference:** Nayeem et al. (2026) [[28](references.md#ref-28)]

### Neural network

#### AudioSeal { #audioseal-method }

`AUDIOSEAL_METHOD` · Neural network · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/AudioSealMethod.py)

*Uses pretrained AudioSeal networks to generate and detect a neural audio watermark.*

Wraps the optional AudioSeal package and released model weights. The generator carries 16 bits per chunk, so longer messages are split across consecutive chunks; alpha scales the watermark contribution. Detection uses the corresponding neural model. Dependencies and weights must be available, and robustness should be measured for the actual audio and attack settings.

| Parameter | Default | Role |
| --- | --- | --- |
| `generator` | `'audioseal_wm_16bits'` |  |
| `detector` | `'audioseal_detector_16bits'` |  |
| `alpha` | `1.0` | embedding strength |
| `min_chunk_length` | `16000` |  |

**Reference:** San Roman et al. (2024) [[36](references.md#ref-36)] · **Requires:** `audioseal`, `torch` — install with `pip install "the-a-files[neural]"`

#### FGAS { #fgas-method }

`FGAS_METHOD` · Neural network · steganography · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/FgasMethod.py)

*Optimises a small audio perturbation so a fixed neural decoder outputs the desired bits.*

Builds a deterministically initialised 1D convolutional decoder from a shared seed and optimises the input perturbation against the target message. The receiver reconstructs that decoder and thresholds its output. epsilon bounds the perturbation; optimisation settings affect cost and success. A small perturbation does not by itself guarantee robustness to codecs or resistance to steganalysis.

| Parameter | Default | Role |
| --- | --- | --- |
| `hidden_channels` | `16` |  |
| `key` | `1337` | secret key |
| `iterations` | `300` |  |
| `epsilon` | `0.02` | embedding strength |
| `learning_rate` | `0.005` |  |
| `alpha` | `50.0` |  |
| `beta` | `0.5` |  |
| `leaky_slope` | `0.2` |  |
| `radius` | `16` |  |

**Reference:** Yan et al. (2025) [[24](references.md#ref-24)] · **Requires:** `tensorflow` — install with `pip install "the-a-files[ai]"`

#### WavMark { #wavmark-method }

`WAVMARK_METHOD` · Neural network · watermarking · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/methods/WavMarkMethod.py)

*Uses the pretrained WavMark neural model to embed and recover bit payloads in audio chunks.*

A one-second model window carries 32 bits: 16 synchronisation bits and 16 payload bits. The decoder searches for the sync pattern; this adapter uses 16 kHz audio and defaults to chunks of at least two seconds to leave search space. Longer messages use consecutive chunks. The package and weights are optional dependencies; robustness depends on the model and evaluated channel.

| Parameter | Default | Role |
| --- | --- | --- |
| `min_chunk_length` | `32000` |  |

**Reference:** Chen et al. (2023) [[37](references.md#ref-37)] · **Requires:** `wavmark`, `torch` — install with `pip install "the-a-files[neural]"`
<!-- /catalogue:methods-details -->
