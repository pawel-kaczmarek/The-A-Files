# Evaluation metrics

The factory registers **25 metrics**. ↑ indicates higher is better; ↓ indicates lower is better.
These are objective predictors or signal diagnostics, not interchangeable perceptual scales.
MOSNet and SRMR rate the processed signal without a clean reference; the remaining metrics compare signals.

| Metric / registry identifier | Direction | Meaning | Source |
| --- | --- | --- | --- |
| **SNR**<br>`SNR_METRIC` | ↑ dB | Ratio of reference energy to squared error energy; sensitive to gain and alignment. | [7](references.md#ref-7) |
| **SNRseg**<br>`SNR_SEG_METRIC` | ↑ dB | Averages framewise signal-to-error ratios, exposing local distortion. | [7](references.md#ref-7) |
| **fwSNRseg**<br>`FWSNR_SEG_METRIC` | ↑ dB | Weights segmental SNR across perceptually motivated frequency bands. | [13](references.md#ref-13) |
| **PESQ**<br>`PESQ_METRIC` | ↑ score | Perceptual speech-quality estimate; accepts 8 or 16 kHz and returns raw narrowband score and MOS-LQO (raw is undefined at 16 kHz). | [8](references.md#ref-8) |
| **WSS**<br>`WSS_METRIC` | ↓ distance | Compares spectral slopes with weights emphasising perceptually salient peaks. | [13](references.md#ref-13) |
| **LLR**<br>`LLR_METRIC` | ↓ distance | Compares LPC spectral envelopes using a log-likelihood ratio. | [13](references.md#ref-13) |
| **CD**<br>`CEPSTRUM_DISTANCE_METRIC` | ↓ dB | Measures differences between LPC-derived cepstral coefficients. | [7](references.md#ref-7) |
| **MCD**<br>`MEL_CEPSTRAL_DISTANCE_METRIC` | ↓ dB | Measures distance between mel-cepstral representations of the signals. | [11](references.md#ref-11) |
| **CSII**<br>`CSII_METRIC` | ↑ index | Uses coherence-based effective SNR to estimate intelligibility at high, middle and low signal levels. | [7](references.md#ref-7) |
| **NCM**<br>`NCM_METRIC` | ↑ index | Combines normalised covariance of band envelopes to predict intelligibility. | [7](references.md#ref-7) |
| **STOI**<br>`STOI_METRIC` | ↑ index | Correlates short-time band envelopes to predict speech intelligibility. | [9](references.md#ref-9) |
| **eSTOI**<br>`ESTOI_METRIC` | ↑ index | Uses normalised short-time spectral patterns to extend intelligibility prediction to modulated interference. | [45](references.md#ref-45) |
| **STGI**<br>`STGI_METRIC` | ↑ index | Estimates intelligibility from preserved spectro-temporal glimpses. | [15](references.md#ref-15) |
| **wSTMI**<br>`WSTMI_METRIC` | ↑ index | Weights spectro-temporal modulation information for intelligibility prediction. | [14](references.md#ref-14) |
| **SRMR**<br>`SRMR_METRIC` | ↑ ratio | Rates modulation-energy distribution without a clean reference; a reverberation-related proxy. | [10](references.md#ref-10) |
| **BSD**<br>`BSD_METRIC` | ↓ distance | Measures Bark-domain spectral distortion; not a direct estimate of room reverberation time. | [7](references.md#ref-7) |
| **Csig**<br>`CSIG_METRIC` | ↑ rating | Regression-based predictor of perceived speech-signal distortion. | [13](references.md#ref-13) |
| **Cbak**<br>`CBAK_METRIC` | ↑ rating | Regression-based predictor of background-noise intrusiveness. | [13](references.md#ref-13) |
| **Covl**<br>`COVL_METRIC` | ↑ rating | Regression-based predictor of overall speech quality. | [13](references.md#ref-13) |
| **SI-SDR**<br>`SISDR_METRIC` | ↑ dB | Separates target projection from residual error after optimal scalar alignment. | [17](references.md#ref-17) |
| **BSSEval v4**<br>`BSS_EVAL_METRIC` | ↑ dB | Returns SDR, ISR, SIR and SAR from a source-separation evaluation; the permutation index is not a quality score. | [18](references.md#ref-18) |
| **MOSNet**<br>`AI_MOSNET_METRIC` | ↑ predicted MOS | Neural no-reference quality predictor developed for voice conversion; domain transfer requires validation. | [16](references.md#ref-16) |
| **LSD**<br>`LSD_METRIC` | ↓ dB | Mean framewise RMS log-magnitude error: 32 ms Hann windows, 75% overlap and −100 dBFS floor. | [TAF definition](research-capabilities.md#measures-and-their-meaning) |
| **MRSC**<br>`MRSC_METRIC` | ↓ ratio | Mean relative STFT magnitude error at 16, 32 and 64 ms; undefined for a silent reference. | [TAF definition](research-capabilities.md#measures-and-their-meaning) |
| **ViSQOL Audio**<br>`VISQOL_METRIC` | ↑ MOS-LQO | Compares spectro-temporal similarity through the official audio model; the adapter resamples metric inputs to 48 kHz. | [46](references.md#ref-46) |

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
