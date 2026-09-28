# Research protocol

Information hiding in audio is governed by four mutually conflicting requirements: **payload capacity**, **perceptual
transparency**, **robustness** to signal processing, and — in the steganographic setting — **statistical
undetectability**. No single method dominates on all four axes, and published results are frequently obtained under
incomparable conditions (different corpora, payload sizes, attack parameters and metric implementations). The A-Files
addresses this by evaluating methods under a shared, recorded experimental protocol.
See the taxonomy and trade-offs in [Alsabhany et al.](references.md#ref-1).

Let `x[n]` denote a cover speech signal sampled at `f_s`, and `b ∈ {0,1}^L` a binary payload of length `L`. An embedding
function `E` produces the stego signal

```
y[n] = E(x[n], b),
```

which may be subjected to a channel or attack operator `A_θ` with explicit parameters `θ`, yielding `z[n] = A_θ(y[n])`.
A blind decoder `D` recovers an estimate `b̂ = D(z[n], L)` without access to the cover. Each trial is characterised by:

* **Reliability** — bit error rate `BER = (1/L) Σ 1[b_i ≠ b̂_i]` and bit accuracy `1 − BER`;
* **Transparency** — objective distortion between `x` and `y` (e.g. SNR, PESQ, STOI), computed *before* the attack so
  that embedding distortion is not confounded with attack damage;
* **Robustness** — BER as a function of the attack family and its severity `θ`;
* **Capacity** — the largest `L` for which decoding remains error-free on a segment of given duration;
* **Detectability** — held-out accuracy of a cover-versus-stego classifier ([steganalysis](steganalysis.md)).

![The A-Files functional overview](functions.svg)

The toolkit comprises:

* 30 embedding implementations and adapters spanning time-domain, transform-domain (DCT, DWT, LWT, SVD, EMD),
  spread-spectrum, quantisation-index-modulation, echo, phase, reversible and neural approaches;
* 25 registered quality/intelligibility metrics, including eSTOI, spectral diagnostics and optional ViSQOL Audio;
* a library of seeded, parameterised attacks grouped by physical phenomenon, with severity presets and composite
  channel pipelines;
* a steganalysis module estimating empirical detectability;
* an asynchronous experiment engine that produces normalised, exportable result tables.

## Signal and payload representation

Audio is represented as a discrete-time waveform with its sampling rate and container metadata. WAV, FLAC and OGG are
supported. Two public speech corpora are bundled as fixed subsets — VCTK (10 utterances) and LibriSpeech (11
utterances) — so that experiments can be repeated without external downloads. Payloads are binary vectors, which
decouples each method's `encode`/`decode` interface from the storage format.

Experiments also accept exact UTF-8 text, hexadecimal bytes, explicit bits and seeded random payloads at requested
bit rates. Rows record exact bit counts, offered bits/s, bits/sample, exact-message goodput, runtime factors,
audio metadata and payload/audio hashes. Exported run configurations pin the selected relative file paths and
SHA-256 digests. The Statistics tab offers filtered research comparisons. See the [research protocol and examples](research-capabilities.md)
for metric definitions, payload conventions, dataset metadata, sources, implementation scope and validation.

## Method contract

Every method is subject to an automated conformance test on real speech from the bundled VCTK subset. The test asserts
that (i) the payload is recovered bit-exactly by a *fresh* decoder instance, so no state is shared between encoder and
decoder; (ii) `encode` neither modifies the caller's cover nor changes its length; (iii) a payload exceeding the
method's capacity raises `CapacityError` (a subclass of `ValueError`, in `taf.models.errors`) instead of being silently
truncated, both just above the capacity of frame-based methods and above four bits per sample, which no packaged method
can carry; and (iv) encoding and decoding are numerically stable on synthetic signals. The experiment engine relies on
(i) and (iii): extraction always runs on a new instance, and `CapacityError` is recorded as an over-capacity outcome
rather than as a crash. Several methods derive embedding strength or quantisation steps from local signal
statistics, such as frame norm or band RMS. Gain robustness depends on the method and processing chain; it
is measured experimentally rather than assumed for the entire catalogue. Optional-model tests may be skipped
when dependencies or weights are unavailable. Conformance checks cover selected inputs and parameter settings.
