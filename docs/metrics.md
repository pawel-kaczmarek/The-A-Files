# Evaluation metrics

These are objective predictors or signal diagnostics, not interchangeable perceptual scales.
MOSNet and SRMR rate the processed signal without a clean reference; the remaining metrics compare signals.

<!-- catalogue:metrics-table -->
**25 metrics** are registered. ↑ indicates higher is better; ↓ indicates lower is better.

| Metric / registry identifier | Direction · scale | What it measures | Reference |
| --- | --- | --- | --- |
| [**BSSEval**](#bss-eval-metric)<br>`BSS_EVAL_METRIC` | ↑ dB | Reports several distortion measures originally designed for source-separation evaluation. | [18](references.md#ref-18) |
| [**Cbak**](#cbak-metric)<br>`CBAK_METRIC` | ↑ 1–5 | Predicts the perceived intrusiveness of background noise in processed speech. | [13](references.md#ref-13) |
| [**CD**](#cepstrum-distance-metric)<br>`CEPSTRUM_DISTANCE_METRIC` | ↓ dB | Compares cepstral representations of reference and processed speech. | [7](references.md#ref-7) |
| [**Covl**](#covl-metric)<br>`COVL_METRIC` | ↑ 1–5 | Predicts overall speech quality, combining signal distortion and noise effects. | [13](references.md#ref-13) |
| [**Csig**](#csig-metric)<br>`CSIG_METRIC` | ↑ 1–5 | Predicts how much the speech signal itself sounds distorted. | [13](references.md#ref-13) |
| [**fwSNRseg**](#fwsnr-seg-metric)<br>`FWSNR_SEG_METRIC` | ↑ dB | Measures segmental SNR with frequency-band weighting relevant to speech. | [13](references.md#ref-13) |
| [**LLR**](#llr-metric)<br>`LLR_METRIC` | ↓ distance | Measures how much the linear-prediction model of speech changes after processing. | [13](references.md#ref-13) |
| [**LSD**](#lsd-metric)<br>`LSD_METRIC` | ↓ dB | Measures the difference between short-time log-magnitude spectra in decibels. | — |
| [**MCD**](#mel-cepstral-distance-metric)<br>`MEL_CEPSTRAL_DISTANCE_METRIC` | ↓ dB | Measures changes in speech spectral shape using a mel-scaled representation. | [11](references.md#ref-11) |
| [**MRSC**](#mrsc-metric)<br>`MRSC_METRIC` | ↓ ratio | Measures relative spectral error at several time-frequency resolutions. | — |
| [**PESQ**](#pesq-metric)<br>`PESQ_METRIC` | ↑ MOS-LQO 1–4.64 | Predicts perceived speech quality by comparing degraded speech with its reference. | [8](references.md#ref-8) |
| [**SI-SDR**](#sisdr-metric)<br>`SISDR_METRIC` | ↑ dB | Measures waveform distortion after compensating for a single global scale factor. | [17](references.md#ref-17) |
| [**SNR**](#snr-metric)<br>`SNR_METRIC` | ↑ dB | Measures waveform fidelity as the ratio of original signal energy to error energy. | [7](references.md#ref-7) |
| [**SNRseg**](#snr-seg-metric)<br>`SNR_SEG_METRIC` | ↑ dB | Averages signal-to-noise ratios over short speech frames. | [7](references.md#ref-7) |
| [**ViSQOL Audio**](#visqol-metric)<br>`VISQOL_METRIC` | ↑ MOS-LQO (audio SVR, maximum about 4.75) | Estimates perceived audio quality from similarity between reference and processed spectrograms. | [46](references.md#ref-46), [46](references.md#ref-46) |
| [**WSS**](#wss-metric)<br>`WSS_METRIC` | ↓ distance | Compares the slopes of reference and processed speech spectra. | [13](references.md#ref-13) |
| [**CSII**](#csii-metric)<br>`CSII_METRIC` | ↑ 0–1 | Estimates intelligibility using coherence between reference and processed speech. | [7](references.md#ref-7) |
| [**eSTOI**](#estoi-metric)<br>`ESTOI_METRIC` | ↑ index (usually 0–1) | Extends STOI with a spectro-temporal comparison useful for fluctuating interference. | [45](references.md#ref-45) |
| [**NCM**](#ncm-metric)<br>`NCM_METRIC` | ↑ 0–1 | Measures preservation of slow speech-envelope modulations across frequency bands. | [7](references.md#ref-7) |
| [**STGI**](#stgi-metric)<br>`STGI_METRIC` | ↑ 0–1 | Estimates how many local spectro-temporal speech patterns remain sufficiently preserved. | [15](references.md#ref-15) |
| [**STOI**](#stoi-metric)<br>`STOI_METRIC` | ↑ 0–1 | Estimates speech intelligibility from preservation of short-time spectral envelopes. | [9](references.md#ref-9) |
| [**wSTMI**](#wstmi-metric)<br>`WSTMI_METRIC` | ↑ index | Predicts intelligibility from weighted spectro-temporal modulation similarities. | [14](references.md#ref-14) |
| [**BSD**](#bsd-metric)<br>`BSD_METRIC` | ↓ distance | Measures reference-relative spectral distortion on the Bark auditory scale. | [7](references.md#ref-7) |
| [**SRMR**](#srmr-metric)<br>`SRMR_METRIC` | ↑ ratio | Estimates reverberation-related degradation from the processed signal alone. | [10](references.md#ref-10) |
| [**MOSNet**](#ai-mosnet-metric)<br>`AI_MOSNET_METRIC` | ↑ MOS 1–5 | Predicts speech quality using a pretrained neural network rather than a samplewise reference comparison. | [16](references.md#ref-16) |
<!-- /catalogue:metrics-table -->

## Reading the results

- **Embedding distortion:** compare cover with stego (`metrics`).
- **Attack damage:** compare stego with attacked audio (`attack_metrics`).
- **Payload recovery:** report BER, exact-message recovery and trial completion separately.
- **Security:** use held-out [steganalysis](steganalysis.md), not a quality score.

Report every component of vector-valued metrics separately. Keep PESQ raw and MOS-LQO values distinct;
do not rank the BSSEval permutation index. STOI, eSTOI, STGI and wSTMI predict speech intelligibility,
not music quality. Composite ratings and MOSNet inherit the domains used to develop their predictors.
The Hu–Loizou study evaluates correlations with listening-test ratings and constructs composite predictors;
it does not establish their validity for every watermarking distortion. [13](references.md#ref-13)

ViSQOL needs separately installed official bindings and its SVR model. MOSNet needs the TensorFlow extra.
Missing dependencies and invalid inputs are metric errors, not valid zero scores. See
[installation](installation.md) and the [measurement definitions](research-capabilities.md#measures-and-their-meaning).

## Metric reference

Generated from the `card` attribute of each class by `python -m taf.catalogue_docs`; edit the card, not this section. The same texts, in English and Polish, appear in the research UI.

<!-- catalogue:metrics-details -->
### Speech quality

#### BSS Eval v4 (BSSEval) { #bss-eval-metric }

`BSS_EVAL_METRIC` · ↑ higher is better · dB · intrusive (compares with the original) · audio material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/BSSEvalMetric.py)

*Reports several distortion measures originally designed for source-separation evaluation.*

Uses museval BSS Eval v4 to decompose estimation errors into target-related, interference and artefact terms. Returns SDR, ISR, SIR and SAR in dB, plus a source permutation index. Higher is better for the four ratios; perm identifies source assignment and is not a quality score. Some components can be degenerate for single-source comparisons.

**Components:** `sdr`, `isr`, `sir`, `sar`, `perm` (reported separately) · **Reference:** Stöter et al. (2018) [[18](references.md#ref-18)]

#### Composite background-noise rating (Cbak) { #cbak-metric }

`CBAK_METRIC` · ↑ higher is better · 1–5 · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/composite/CbakMetric.py)

*Predicts the perceived intrusiveness of background noise in processed speech.*

Combines PESQ, segmental SNR and weighted spectral slope through an empirical regression. The resulting composite score is limited to the 1–5 range, with higher values meaning less intrusive background noise. It is calibrated for speech enhancement and needs the original reference; it is not a direct measurement of background-noise power.

**Reference:** Hu & Loizou (2008) [[13](references.md#ref-13)]

#### Cepstral distance (CD) { #cepstrum-distance-metric }

`CEPSTRUM_DISTANCE_METRIC` · ↓ lower is better · dB · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/CepstrumDistanceMetric.py)

*Compares cepstral representations of reference and processed speech.*

Derives cepstral coefficients from short-frame speech models and computes their distance. Cepstral features describe spectral-envelope shape rather than individual waveform samples. Lower distance in dB means a closer envelope. It needs a reference and should be interpreted alongside other metrics, since matching envelopes do not imply identical phase or perceptual quality.

**Reference:** Loizou (2013) [[7](references.md#ref-7)]

#### Composite overall-quality rating (Covl) { #covl-metric }

`COVL_METRIC` · ↑ higher is better · 1–5 · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/composite/CovlMetric.py)

*Predicts overall speech quality, combining signal distortion and noise effects.*

Uses an empirical combination of PESQ, log-likelihood ratio and weighted spectral slope to produce a composite 1–5 score. Higher is better. The model was designed for speech-enhancement evaluation; its scale should not be treated as directly interchangeable with PESQ, MOSNet or a listening-test MOS.

**Reference:** Hu & Loizou (2008) [[13](references.md#ref-13)]

#### Composite signal-distortion rating (Csig) { #csig-metric }

`CSIG_METRIC` · ↑ higher is better · 1–5 · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/composite/CsigMetric.py)

*Predicts how much the speech signal itself sounds distorted.*

Combines PESQ, log-likelihood ratio and weighted spectral slope using an empirical regression, then limits the prediction to 1–5. Higher values mean better predicted speech-signal quality. It focuses on speech distortion rather than background-noise annoyance and is a model estimate, not a listener’s actual rating.

**Reference:** Hu & Loizou (2008) [[13](references.md#ref-13)]

#### Frequency-weighted segmental SNR (fwSNRseg) { #fwsnr-seg-metric }

`FWSNR_SEG_METRIC` · ↑ higher is better · dB · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/FWSnrSegMetric.py)

*Measures segmental SNR with frequency-band weighting relevant to speech.*

Separates short-time spectra into perceptual bands, computes bandwise signal-to-error ratios and combines them with frequency-dependent weights. Higher scores indicate less weighted distortion. It can distinguish errors with similar total energy but different spectral locations; it requires a reference and temporal alignment.

**Reference:** Hu & Loizou (2008) [[13](references.md#ref-13)]

#### Log-likelihood ratio (LLR) { #llr-metric }

`LLR_METRIC` · ↓ lower is better · distance · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/LlrMetric.py)

*Measures how much the linear-prediction model of speech changes after processing.*

Fits linear predictive coefficients to reference and processed frames, then evaluates a log ratio of prediction-error terms using the reference autocorrelation. Lower values indicate more similar spectral envelopes. It is intended for speech; silence, numerical conditioning and time alignment can affect the estimate.

**Reference:** Hu & Loizou (2008) [[13](references.md#ref-13)]

#### Log-spectral distance (LSD) { #lsd-metric }

`LSD_METRIC` · ↓ lower is better · dB · intrusive (compares with the original) · audio material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/LogSpectralDistanceMetric.py)

*Measures the difference between short-time log-magnitude spectra in decibels.*

Uses 32 ms Hann windows with 75% overlap and a −100 dBFS magnitude floor. For each frame it takes the RMS spectral difference in dB, then averages frames. Lower is closer to the reference. It exposes spectral changes but ignores phase differences that leave magnitudes unchanged; it is a diagnostic, not a MOS predictor.

**Reference:** —

#### Mel-cepstral distance (MCD) { #mel-cepstral-distance-metric }

`MEL_CEPSTRAL_DISTANCE_METRIC` · ↓ lower is better · dB · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/MelCepstralDistanceMetric.py)

*Measures changes in speech spectral shape using a mel-scaled representation.*

Builds 20-band mel power spectrograms with a 1024-sample Hamming window and 256-sample hop, then passes them to the mel-cepstral-distance comparison library. Lower MCD means closer representations. Values depend on feature extraction and comparison settings, so scores from differently configured MCD implementations are not automatically interchangeable.

**Reference:** Kubichek (1993) [[11](references.md#ref-11)]

#### Multi-resolution spectral convergence (MRSC) { #mrsc-metric }

`MRSC_METRIC` · ↓ lower is better · ratio · intrusive (compares with the original) · audio material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/SpectralConvergenceMetric.py)

*Measures relative spectral error at several time-frequency resolutions.*

Computes STFT magnitudes with 16, 32 and 64 ms windows and 75% overlap. At each resolution it divides the Frobenius norm of the magnitude error by the reference norm, then averages. Lower is better and zero indicates matching magnitudes. A silent reference is undefined; some phase changes remain invisible. This ratio is not a perceptual MOS.

**Reference:** —

#### Perceptual evaluation of speech quality (PESQ) { #pesq-metric }

`PESQ_METRIC` · ↑ higher is better · MOS-LQO 1–4.64 · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/PesqMetric.py)

*Predicts perceived speech quality by comparing degraded speech with its reference.*

Uses the PESQ perceptual model to align and compare auditory representations, combining disturbance measures into MOS-LQO. Higher is better; the catalogue reports the wideband scale up to about 4.64. The implementation depends on the PESQ library and supported speech sample rates. It is a speech-quality estimate, not a general music metric or a bit-error measure. Input must be sampled at 8 or 16 kHz. Two components are reported: the raw narrowband P.862 score (undefined, and reported as missing, at 16 kHz) and MOS-LQO.

**Components:** `p862_raw`, `mos_lqo` (reported separately) · **Reference:** ITU-T P.862 / Wang et al. (2022) [[8](references.md#ref-8)]

#### Scale-invariant signal-to-distortion ratio (SI-SDR) { #sisdr-metric }

`SISDR_METRIC` · ↑ higher is better · dB · intrusive (compares with the original) · audio material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/SisdrMetric.py)

*Measures waveform distortion after compensating for a single global scale factor.*

Projects the processed signal onto the reference and compares projected target energy with residual energy in dB. Higher SI-SDR is better. Unlike ordinary SNR, a uniform gain change is discounted; timing errors and other waveform changes still count. Use alongside level-sensitive metrics when amplitude preservation matters.

**Reference:** Le Roux et al. (2019) [[17](references.md#ref-17)]

#### Signal-to-noise ratio (SNR) { #snr-metric }

`SNR_METRIC` · ↑ higher is better · dB · intrusive (compares with the original) · audio material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/SnrMetric.py)

*Measures waveform fidelity as the ratio of original signal energy to error energy.*

Computes 10 log10 of reference energy divided by the energy of the samplewise difference. Higher dB means less error relative to the reference. Requires aligned signals and penalises gain or time shifts strongly. It measures numerical fidelity, not perceived quality, message recovery or steganographic detectability.

**Reference:** Loizou (2013) [[7](references.md#ref-7)]

#### Segmental signal-to-noise ratio (SNRseg) { #snr-seg-metric }

`SNR_SEG_METRIC` · ↑ higher is better · dB · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/SnrSegMetric.py)

*Averages signal-to-noise ratios over short speech frames.*

Calculates a local energy-to-error ratio for each frame, limits extreme frame scores and averages them. This gives quieter sections more influence than a single global SNR. Higher dB is better. Frame length, silence and alignment affect the result; it remains an error-energy measure rather than a listening score.

**Reference:** Loizou (2013) [[7](references.md#ref-7)]

#### ViSQOL Audio (ViSQOL Audio) { #visqol-metric }

`VISQOL_METRIC` · ↑ higher is better · MOS-LQO (audio SVR, maximum about 4.75) · intrusive (compares with the original) · audio material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/VisqolMetric.py)

*Estimates perceived audio quality from similarity between reference and processed spectrograms.*

Uses official Google ViSQOL bindings and the audio-mode SVR model, with both inputs polyphase-resampled to 48 kHz. Spectro-temporal similarity is mapped to MOS-LQO, whose audio-model maximum is about 4.75. Prefer active clips around 8–10 seconds. Requires optional bindings and model data; do not compare these scores directly with ViSQOL speech mode.

**Reference:** Chinen et al. (2020) [[46](references.md#ref-46)], google/visqol [[46](references.md#ref-46)] · **Requires:** `visqol`

#### Weighted spectral slope (WSS) { #wss-metric }

`WSS_METRIC` · ↓ lower is better · distance · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_quality/WssMetric.py)

*Compares the slopes of reference and processed speech spectra.*

Calculates differences between adjacent perceptual-band levels and weights slope mismatches, emphasising important spectral peaks. Lower distance means better preservation of spectral shape. Requires reference speech and aligned frames. It highlights spectral-envelope damage but does not directly measure intelligibility or payload recovery.

**Reference:** Hu & Loizou (2008) [[13](references.md#ref-13)]

### Speech intelligibility

#### Coherence speech intelligibility index (CSII) { #csii-metric }

`CSII_METRIC` · ↑ higher is better · 0–1 · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_intelligibility/CsiiMetric.py)

*Estimates intelligibility using coherence between reference and processed speech.*

Computes coherence-based signal-to-distortion information in weighted frequency bands. Returns separate high, mid and low indices for speech segments at different levels. Higher values indicate better preserved intelligibility cues. The three components describe different portions of speech and should not be mistaken for repeated measurements of one global score.

**Components:** `high`, `mid`, `low` (reported separately) · **Reference:** Loizou (2013) [[7](references.md#ref-7)]

#### Extended short-time objective intelligibility (eSTOI) { #estoi-metric }

`ESTOI_METRIC` · ↑ higher is better · index (usually 0–1) · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_intelligibility/EstoiMetric.py)

*Extends STOI with a spectro-temporal comparison useful for fluctuating interference.*

Uses pystoi in extended mode, comparing normalised short-time time-frequency envelope patterns. Higher eSTOI indicates better predicted speech intelligibility, usually on an index scale near 0–1. It requires reference speech and remains a proxy: it does not measure listening quality, message decoding or steganographic security.

**Reference:** Jensen & Taal (2016) [[45](references.md#ref-45)]

#### Normalised-covariance measure (NCM) { #ncm-metric }

`NCM_METRIC` · ↑ higher is better · 0–1 · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_intelligibility/NcmMetric.py)

*Measures preservation of slow speech-envelope modulations across frequency bands.*

Splits speech into 20 bands, extracts Hilbert envelopes and resamples them to 32 Hz. Normalised envelope covariance is mapped to apparent SNR and combined using speech-importance weights. Higher scores indicate better predicted intelligibility. This implementation accepts 8 or 16 kHz input and requires a reference; it does not assess bit recovery.

**Reference:** Loizou (2013) [[7](references.md#ref-7)]

#### Spectro-temporal glimpsing index (STGI) { #stgi-metric }

`STGI_METRIC` · ↑ higher is better · 0–1 · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_intelligibility/StgiMetric.py)

*Estimates how many local spectro-temporal speech patterns remain sufficiently preserved.*

Resamples speech to 10 kHz, builds log-mel spectrograms and applies spectro-temporal Gabor filters. Normalised local similarities are compared with channel-specific thresholds; the score averages the resulting preserved-pattern decisions. Higher is better on a 0–1 scale. Sufficient speech duration is needed for the analysis windows; the result is not a word-recognition rate.

**Reference:** Edraki et al. (2021) [[15](references.md#ref-15)]

#### Short-time objective intelligibility (STOI) { #stoi-metric }

`STOI_METRIC` · ↑ higher is better · 0–1 · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_intelligibility/StoiMetric.py)

*Estimates speech intelligibility from preservation of short-time spectral envelopes.*

Compares reference and processed temporal envelopes in one-third-octave bands over short segments, after the algorithm’s normalisation and clipping steps. Higher values, commonly near the 0–1 range, suggest better intelligibility. It does not recognise words: the score is not a percentage of correctly understood words and is not a general music-quality metric.

**Reference:** Taal et al. (2010) [[9](references.md#ref-9)]

#### Weighted spectro-temporal modulation index (wSTMI) { #wstmi-metric }

`WSTMI_METRIC` · ↑ higher is better · index · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_intelligibility/WstmiMetric.py)

*Predicts intelligibility from weighted spectro-temporal modulation similarities.*

Computes log-mel features and Gabor modulation responses after conversion to 10 kHz. Normalised reference-to-processed correlations are combined with fixed learned weights and a bias. Higher values indicate better predicted intelligibility. This is an index, not a percentage or MOS; interpret it consistently within the same implementation and speech domain.

**Reference:** Edraki et al. (2021) [[14](references.md#ref-14)]

### Reverberation

#### Bark spectral distortion (BSD) { #bsd-metric }

`BSD_METRIC` · ↓ lower is better · distance · intrusive (compares with the original) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_reverberation/BsdMetric.py)

*Measures reference-relative spectral distortion on the Bark auditory scale.*

Maps short-time power spectra into 32 Bark bands and averages the squared band-energy error normalised by reference band energy. Lower values mean closer spectra. Although grouped with reverberation metrics, it responds to other spectral changes as well and is not a direct measurement of room reverberation time.

**Reference:** Loizou (2013) [[7](references.md#ref-7)]

#### Speech-to-reverberation modulation energy ratio (SRMR) { #srmr-metric }

`SRMR_METRIC` · ↑ higher is better · ratio · non-intrusive (rates the signal alone) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/speech_reverberation/SrmrMetric.py)

*Estimates reverberation-related degradation from the processed signal alone.*

Analyses modulation energy in auditory-band envelopes and forms a ratio of lower to higher modulation-frequency energy. Higher SRMR generally suggests less reverberant degradation for speech. It does not require a clean reference and is not a direct RT60 estimate. Noise, speaking style and signal content can also influence its value.

**Components:** `cover`, `processed` (reported separately) · **Reference:** Falk et al. (2010) [[10](references.md#ref-10)]

### Learned (AI-based)

#### MOSNet (MOSNet) { #ai-mosnet-metric }

`AI_MOSNET_METRIC` · ↑ higher is better · MOS 1–5 · non-intrusive (rates the signal alone) · speech material · [source code](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/src/taf/metrics/ai_based/mosnet/MosNetMetric.py)

*Predicts speech quality using a pretrained neural network rather than a samplewise reference comparison.*

Feeds magnitude spectrograms through convolutional and bidirectional LSTM layers and averages frame predictions. TAF evaluates cover and processed audio independently, returning both scores; only processed is ranked as the result. Higher predicts better quality. Requires TensorFlow and model weights; predictions depend on the training domain and are not actual listener ratings. MOSNet was developed for voice conversion; applying it to watermarked or attacked audio requires validation.

**Components:** `cover`, `processed` (reported separately) · **Reference:** Lo et al. (2019) [[16](references.md#ref-16)] · **Requires:** `tensorflow` — install with `pip install "the-a-files[ai]"`
<!-- /catalogue:metrics-details -->
