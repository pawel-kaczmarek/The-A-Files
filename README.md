# The A-Files

![logo.png](docs/logo.png)

> **The A-Files** (`taf`) is an open-source research toolkit for the reproducible evaluation of audio steganography and
> watermarking methods applied to speech. It provides reference implementations of embedding schemes, objective
> measures of perceptual transparency and intelligibility, a parameterised and seeded model of channel distortions and
> removal attacks, and a statistical steganalysis procedure for estimating detectability.

🇵🇱 A Polish-language guide to the structure and content of this document is available in
[docs/README.pl.md](docs/README.pl.md).

## Table of contents

1. [Scope and problem formulation](#about)
2. [Installation](#install)
3. [Usage](#usage)
4. [Experiment engine, REST API and web platform](#platform)
    1. [Extension through plugins](#plugins)
5. [Steganography and watermarking methods](#steganography-algorithms)
6. [Objective quality metrics](#metrics)
    1. [Data-driven (AI-based) metrics](#ai-based)
    2. [Speech reverberation](#speech-reverberation)
    3. [Speech intelligibility](#speech-intelligibility)
    4. [Speech quality](#speech-quality)
7. [Steganalysis](#steganalysis)
8. [Attack and channel model](#attacks)
9. [References](#references)
    1. [Literature](#articles)
    2. [Software resources](#links)
10. [Licence](#licence)
11. [External dependencies](#dependencies)
12. [Authors](#authors)

<a id="about"></a>

## 1. Scope and problem formulation

Information hiding in audio is governed by four mutually conflicting requirements: **payload capacity**, **perceptual
transparency**, **robustness** to signal processing, and — in the steganographic setting — **statistical
undetectability**. No single method dominates on all four axes, and published results are frequently obtained under
incomparable conditions (different corpora, payload sizes, attack parameters and metric implementations). The A-Files
addresses this by evaluating all methods under a single, fully specified experimental protocol.

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
* **Detectability** — held-out accuracy of a cover-versus-stego classifier (Section [7](#steganalysis)).

<img src="docs/functions.svg" alt="The A-Files functional overview">

The toolkit comprises:

* reference implementations of 27 embedding methods spanning time-domain, transform-domain (DCT, DWT, LWT, SVD),
  spread-spectrum, quantisation-index-modulation, echo, phase and neural approaches;
* 21 objective metrics of speech quality, intelligibility and reverberation, including a learned MOS predictor;
* a library of seeded, parameterised attacks grouped by physical phenomenon, with severity presets and composite
  channel pipelines;
* a steganalysis module estimating empirical detectability;
* an asynchronous experiment engine that produces normalised, exportable result tables.

###### Signal and payload representation

Audio is represented as a discrete-time waveform with its sampling rate and container metadata. WAV, FLAC and OGG are
supported. Two public speech corpora are bundled as fixed subsets — VCTK (10 utterances) and LibriSpeech (11
utterances) — so that experiments can be repeated without external downloads. Payloads are binary vectors, which
decouples each method's `encode`/`decode` interface from the storage format.

###### Method contract

Every method is subject to an automated conformance test on real speech from the bundled VCTK subset. The test asserts
that (i) the payload is recovered bit-exactly by a *fresh* decoder instance, so no state is shared between encoder and
decoder; (ii) `encode` neither modifies the caller's cover nor changes its length; (iii) a payload exceeding the
method's capacity raises `CapacityError` (a subclass of `ValueError`, in `taf.models.errors`) instead of being silently
truncated, both just above the capacity of frame-based methods and above four bits per sample, which no packaged method
can carry; and (iv) encoding and decoding are numerically stable on synthetic signals. The experiment engine relies on
(i) and (iii): extraction always runs on a new instance, and `CapacityError` is recorded as an over-capacity outcome
rather than as a crash. Embedding strengths and quantisation steps are defined relative to signal
quantities the decoder can recompute (frame norm, band RMS, mean amplitude), which makes the methods invariant to global
gain and usable on low-level recordings.

<a id="install"></a>

## 2. Installation

The package is distributed on PyPI:

```bash
pip install the-a-files
```

Optional components are provided as extras:

| Extra | Command | Enables |
| --- | --- | --- |
| `neural` | `pip install "the-a-files[neural]"` | Pretrained neural watermarking baselines (`AudioSealMethod`, `WavMarkMethod`; PyTorch) |
| `ai` | `pip install "the-a-files[ai]"` | `FgasMethod` and `MosNetMetric` (TensorFlow ≥ 2.15) |
| `experiments` | `pip install "the-a-files[experiments]"` | Experiment engine (pandas) |
| `platform` | `pip install "the-a-files[platform]"` | Research platform: REST API with PostgreSQL persistence and the corpus library (FastAPI, SQLAlchemy, Alembic, psycopg) |
| `dev` | `pip install -e ".[dev]"` | Test and build tooling (pytest, build, twine) |

See [External dependencies](#dependencies) for system-level prerequisites (C++ build tools, FFmpeg).

<a id="usage"></a>

## 3. Usage

The bundled evaluation workflow is exposed through the `taf-eval` entry point, which accepts the name of a packaged
scenario or a path to a YAML configuration:

```bash
taf-eval direct-no-metrics   # embedding and decoding only
taf-eval full                # embedding, attacks and all metrics
```

Individual components are available through registry-based factories:

```python
from taf.methods.factory import SteganographyMethodFactory
from taf.models.types import MethodType

method = SteganographyMethodFactory.get(16000, MethodType.LSB_METHOD)
stego = method.encode(cover, message)
decoded = method.decode(stego, len(message))
```

<a id="platform"></a>

## 4. Experiment engine, REST API and web platform

### 4.1 Experiment engine

The `taf.experiments` module defines an experiment declaratively and executes it without any user interface:

```python
from taf.experiments import ExperimentConfig, ExperimentType, run_experiment

config = ExperimentConfig(
    experiment_type=ExperimentType.DATASET_BENCHMARK,
    name="lsb-vs-fsvc-vctk",
    dataset_id="vctk",            # example | vctk | librispeech | all (library:<id> within the platform)
    file_limit=4,                 # dataset_path="C:/my/corpus" is also accepted
    methods=["LSB_METHOD", "FSVC_METHOD"],
    metrics=["SNR_METRIC", "PESQ_METRIC"],
    attacks=["additive_noise"],   # each attack is paired with a no-attack baseline
    payload_lengths=[16, 32],
    repetitions=3,
    random_seed=42,
)
run = run_experiment(config)      # per-row failures are recorded, never raised
print(run.status, run.summary["overall"])
run.to_csv("detailed_results.csv")
```

Nine experimental designs share one configuration schema. Each answers one research question about one of the four
properties of an information-hiding system:

| Property | Design | Question | Primary analysis |
| --- | --- | --- | --- |
| Imperceptibility | `perceptual_quality` | How much does embedding degrade the audio? | Per-metric ranks, mean rank, Friedman + Holm–Wilcoxon |
| Robustness | `attack_robustness` | Which method survives a battery of attacks? | Method × attack BER matrix, paired comparison per attack |
| Robustness | `robustness_curve` | At which attack strength does each method break down? | Dose–response curve of BER with bands, breakdown point |
| Capacity | `embedding_capacity` | What is the largest payload a method carries reliably? | Per-file capacity (bits, bit/s), bootstrap interval |
| Security | `detectability` | Can a steganalyser tell stego from cover? | Ensemble steganalysis, Wilson interval, binomial test |
| Multi-criteria | `tradeoff_curve` | How does embedding strength trade transparency for robustness? | Quality–BER curve over a method parameter, Pareto front |
| Multi-criteria | `method_comparison` | Which methods are not beaten on every criterion at once? | Pareto front, paired tests per criterion |
| Multi-criteria | `dataset_benchmark` | How do methods perform across a dataset? | Summary per method, clean and under attack |
| — | `research_experiment` | Any combination of factors (exploratory) | Every analysis that applies |

All designs except `detectability` execute a full factorial design (files × methods × payload lengths × repetitions ×
attack variants, each attack next to a no-attack baseline) and differ in the factors they fix and the analysis applied.
Their results are normalised rows: file, method with its parameters, payload in bits and bits per second, attack with
its resolved parameters, BER, bit accuracy, metric values, timings, status and failure kind. The `detectability` design
runs the steganalysis of Section [7](#steganalysis) for every method and payload length.

Methods are named by catalogue name or by a specification that sets constructor parameters, e.g.
`"QIM_METHOD:step_scale=0.1"`; the label and the parameters are recorded in every row. The two curve designs generate
their conditions from one parameter:

```python
from taf.experiments.sweeps import ParameterSweep

# robustness_curve: one attack parameter from mild to harsh, read in the given order
attack_sweep = ParameterSweep(target="awgn", parameter="snr_db", values=[40, 30, 20, 15, 10, 5, 0])
# tradeoff_curve: one method parameter, usually the one the catalogue marks as the embedding strength
method_sweep = ParameterSweep(target="QIM_METHOD", parameter="step_scale", values=[0.025, 0.05, 0.1, 0.2, 0.4])
```

The breakdown point of a robustness curve is where, in sweep order, the BER first exceeds the usable threshold (0.10),
linearly interpolated between the last usable and the first unusable setting; it is reported as `never` or `always`
when the curve does not cross.

#### Experimental protocol

* **Randomness.** A single experiment seed determines every random quantity. When none is given, one is drawn and
  stored in the exported configuration. Messages are seeded by their length, so messages of different lengths are
  independent draws and adding a payload length leaves the others unchanged. Attack realisations are seeded by file,
  repetition and attack, but not by method: repetitions sample the channel independently, while every method meets
  the same noise realisation within a trial (common random numbers), which keeps comparisons between methods paired.
* **Blind extraction.** Decoding runs on a fresh method instance that has seen neither the cover nor the message.
* **Separation of effects.** Cover-to-stego metrics (imperceptibility) are computed once per embedded signal and are
  never mixed with stego-to-attacked metrics (attack damage). Multi-valued metrics are reported per named component.
* **Failures are not bit errors.** A trial that yields no decoded message is labelled `over_capacity`, `encode_error`,
  `io_error`, `attack_error` or `decode_error`, and has no BER. Bit-level statistics use completed trials and are
  reported together with the completion rate. Where one robustness figure must include failures, they are scored at
  chance level (BER = 0.5).

#### Statistical analysis

The file is the unit of replication, because rows of one file (payloads, repetitions, attacks) are correlated.

* **Uncertainty.** 95% intervals come from a cluster bootstrap that resamples whole files.
* **Method comparison.** Methods are compared on per-file means in a paired design. Two methods are compared with the
  Wilcoxon signed-rank test; more than two with the Friedman test followed by Holm-corrected pairwise Wilcoxon tests,
  rank-biserial effect sizes and the Nemenyi critical difference (Demšar, 2006). Below a certain number of files, no
  Holm-corrected pairwise test can reach p < 0.05, whatever the data: 6 files for two methods, 7 for three and 14 for
  all 27. The preview warns when the dataset is smaller than that.
* **Multi-criteria summary.** Methods are summarised by their Pareto front over bit accuracy, robustness and every
  metric with a declared direction. A weighted score is computed only when weights are supplied, because it depends on
  the weights and, through normalisation, on the set of compared methods.
* **Capacity.** Capacity is determined per file as the largest tested payload below the first failing one, where a
  payload passes when all trials complete and the bit-accuracy and BER thresholds are met. It is reported in bits and in
  bits per second, and summarised across files with a bootstrap interval. Payloads of up to 8192 bits can be tested.

#### Evaluation block

Next to the analysis of its design, every factorial run carries a shared evaluation block
(`taf.experiments.scenarios.evaluation`, `summary["evaluation"]`), computed with the same estimators:

* **Per method**, without and under attack: mean with its interval, median, SD, range, quartiles and IQR of BER over
  completed trials; Tukey box summaries; recovery rates at BER = 0, ≤ 1 % and ≤ 5 % (failed trials count as not
  recovered). With *L*-bit payloads BER moves in steps of 1/*L*, so below 100 bits "≤ 1 %" can only mean BER = 0; the
  block records this.
* **Per attack**: the BER increase it causes, as the mean over files of attacked minus clean BER (each file paired with
  itself), the methods ordered by their BER under it, and paired comparisons of the methods.
* **Per attack family observed at several strengths** (severity levels or a swept parameter): the degradation curve
  and a test of monotonic trend.
* **Quality** cover-vs-stego kept apart from attack damage stego-vs-attacked; **payload** (bits, bit/s) against BER and
  quality; **processing time** and real-time factor, compared between methods only when measured with one worker.
* **Correlations** (payload vs BER, quality and time; attack strength vs BER) as Spearman ρ on per-(file, level) means,
  with a permutation test that shuffles values *within* files, a bootstrap interval over files, and Holm correction
  over all correlation tests of the run.
* **Findings**: statements computed from these figures (largest BER increase with its interval, methods whose upper
  95 % bound of BER stays ≤ 0.01 under an attack, failures, between-file variability, significant trends, outcome of
  the paired tests with the largest significant effect, and when too few files make significance unreachable). They
  also open the LaTeX and Markdown reports.

Confidence level, bootstrap resamples and seed, tests, effect size, correction, permutations and seed are stored with
the block, so every figure can be recomputed from the stored trials.

#### Provenance

Every run produces a manifest (`ExperimentRun.manifest`, `GET /api/runs/{id}/manifest.json`). It records the
package version and source commit (with a flag for uncommitted changes), the Python, platform, numerical-library and
FFmpeg versions, the SHA-256 digest, sampling rate and duration of every input file, the resolved seed and the
expanded attack list. It also states whether timings are comparable: timings are comparable only with
`max_workers = 1`, and speed is otherwise excluded from method comparisons.

#### Reproduction of single trials and reports

Because every random quantity is derived from the run seed, any stored trial can be re-synthesised exactly
(`taf.experiments.inspector.resynthesize`): cover, stego, attacked signal and residual (stego − cover), with the decoded
message checked against the stored one. `taf.experiments.reporting.build_report` summarises a run as a LaTeX (booktabs)
or Markdown report: an *Experimental setup* paragraph generated from the configuration and manifest, the result tables
with 95% intervals, and the statistical tests.

### 4.2 Research platform: persistence, REST API and web client

The platform stores experiments, runs, result rows and datasets in PostgreSQL. An *experiment* is a versioned protocol
(title, research question, hypothesis, configuration); a *run* executes one version and keeps a snapshot of the
configuration, the summary and the manifest, so results remain interpretable after the protocol changes.

```bash
docker compose up -d db          # PostgreSQL 17 on localhost:5432 (user, password and database: taf)
pip install "the-a-files[platform]"
taf-api                          # http://127.0.0.1:8000 — OpenAPI documentation at /docs
```

Migrations (Alembic) are applied when the API starts. `TAF_DATABASE_URL` overrides the connection string
(`postgresql+psycopg://taf:taf@localhost:5432/taf`), `TAF_DATA_DIR` (default `~/.taf`) holds prepared corpora and
uploads, and `TAF_MAX_CONCURRENT_RUNS` limits parallel runs.

| Purpose | Endpoints |
| --- | --- |
| Catalogue | `GET /api/catalog/{methods,metrics,attacks,designs,presets,corpora,datasets}` |
| Experiments | `GET, POST /api/experiments`, `GET, PUT, DELETE /api/experiments/{id}`, `POST /api/experiments/{id}/{duplicate,archive}`, `POST /api/experiments/preview` |
| Runs | `POST /api/experiments/{id}/runs`, `GET /api/runs`, `GET /api/runs/{id}/{summary,rows,facets,manifest.json,config.json}`, `POST /api/runs/{id}/cancel` |
| Exports and reports | `GET /api/runs/{id}/{export.csv,export_summary.csv,report.tex,report.md}` |
| Trial inspector | `GET /api/runs/{id}/rows/{row}/inspect`, `GET /api/runs/{id}/rows/{row}/audio/{cover,stego,attacked,residual}.wav` |
| Progress (Server-Sent Events) | `GET /api/runs/events`, `GET /api/runs/{id}/events` |
| Datasets | `GET /api/datasets`, `POST /api/datasets/{prepare,local,upload,synthetic}`, `GET, DELETE /api/datasets/{id}` |

The web client in [`web/`](web/README.md) (Next.js, TypeScript; English and Polish; light and dark theme) is a thin
client of this API. It offers a design gallery grouped by property, a step-by-step protocol editor with an execution
plan, live runs, result figures with table views, critical-difference diagrams, the trial inspector and report
downloads. It is not part of the PyPI distribution:

```bash
cd web
npm install
npm run dev   # http://localhost:3000
```

#### Evaluation corpora

`taf.corpora` catalogues standard corpora with licence, citation and DOI: speech (LibriSpeech, Mini LibriSpeech,
VCTK 0.92, TSP, LJSpeech, LibriTTS-R, EARS, TIMIT, Common Voice, NOIZEUS), synthetic speech (ASVspoof 5, MLAAD), music
(MUSDB18-HQ, GTZAN) and environmental sound (ESC-50). Corpora with an open download are prepared on the server as
reproducible subsets by a recorded rule: a seeded selection balanced across speakers, conversion to mono, polyphase
resampling to the target rate, optional excerpts, 16-bit FLAC, and a manifest with the SHA-256 digest of the archive and
of every file. Licensed corpora are registered from a local copy. Uploads, local directories and a set of synthetic test
signals (tones, sweep, white and pink noise, tone bursts, square wave, low-level noise) complete the library. The
packaged VCTK and LibriSpeech subsets remain available without download.

<a id="plugins"></a>

### 4.3 Extension through plugins

Methods, metrics and attacks from other distributions are discovered through entry points, without modifying this
package. A registered component appears in the catalogue, in the API and in experiments under its entry-point name, and
is subject to the same method contract:

```toml
# pyproject.toml of the distribution that provides the components
[project.entry-points."taf.methods"]
MY_METHOD = "my_package.method:MyMethod"      # callable taking the sampling rate

[project.entry-points."taf.metrics"]
MY_METRIC = "my_package.metric:MyMetric"      # callable taking no arguments

[project.entry-points."taf.attacks"]
my_attack = "my_package.attack:MyAttack"      # Attack subclass
```

Packaged names take precedence, so a published result that names a packaged method always refers to the packaged
implementation (`taf.plugins`). An automated test enforces the package layering: the building blocks (`methods`,
`metrics`, `attacks`, `models`, `audio`, `steganalysis`) never import the evaluation engine, the experiment layer or the
HTTP layer. The engine in turn never imports the HTTP layer.

<a id="steganography-algorithms"></a>

## 5. Steganography and watermarking methods

Table 1 lists the implemented methods together with the publications on which they are based.

**Table 1.** Implemented embedding methods.

| No. | Module                                      | Method                                                                  | Ref.              |
|-----|---------------------------------------------|-------------------------------------------------------------------------|-------------------|
| 1.  | `LsbMethod.py`                              | Least-significant-bit substitution                                      | [[1]](#articles)  |
| 2.  | `EchoMethod.py`                             | Echo hiding with a single echo kernel                                   | [[1]](#articles)  |
| 3.  | `PhaseCodingMethod.py`                      | Phase coding                                                            | [[1]](#articles)  |
| 4.  | `ImprovedPhaseCodingMethod.py`              | Improved phase coding                                                   | [[19]](#articles) |
| 5.  | `DctDeltaLsbMethod.py`                      | DCT delta LSB embedding                                                 | [[1]](#articles)  |
| 6.  | `DwtLsbMethod.py`                           | DWT-domain LSB embedding                                                | [[1]](#articles)  |
| 7.  | `DctB1Method.py`                            | First-band DCT coefficient embedding (DCT-b1)                           | [[2]](#articles)  |
| 8.  | `PatchworkMultilayerMethod.py`              | Patchwork-based multilayer watermarking                                 | [[3]](#articles)  |
| 9.  | `NormSpaceMethod.py`                        | Norm-space watermarking                                                 | [[4]](#articles)  |
| 10. | `FsvcMethod.py`                             | Frequency singular value coefficient modification (FSVC)                | [[5]](#articles)  |
| 11. | `DsssMethod.py`                             | Direct-sequence spread spectrum (DSSS)                                  | [[6]](#articles)  |
| 12. | `BlindSvdMethod.py`                         | Blind SVD embedding with entropy and log-polar transformation           | [[20]](#articles) |
| 13. | `PrimeFactorInterpolatedMethod.py`          | Least-prime-factor interpolated embedding                               | [[21]](#articles) |
| 14. | `LwtMethod.py`                              | Lifting wavelet transform (LWT) embedding                               | [[22]](#articles) |
| 15. | `ForegroundBackgroundSegmentationMethod.py` | Foreground-background segmentation LSB (FBS-LSB)                        | [[23]](#articles) |
| 16. | `FgasMethod.py`                             | Fixed-decoder network with adversarial perturbation generation (FGAS)   | [[24]](#articles) |
| 17. | `AacStcMethod.py`                           | Adaptive ±1 LSB with AAC perceptual residual and syndrome-trellis codes | [[25]](#articles) |
| 18. | `WirelessDwtLsbMethod.py`                   | DWT-LSB embedding for wireless channels                                 | [[27]](#articles) |
| 19. | `LearnableEmbeddingGaMethod.py`             | Learnable embedding with genetic optimisation                           | [[28]](#articles) |
| 20. | `QimMethod.py`                              | Quantisation index modulation, spread-transform dither modulation       | [[29]](#articles) |
| 21. | `ImprovedSpreadSpectrumMethod.py`           | Improved spread spectrum with host-interference rejection               | [[30]](#articles) |
| 22. | `BackwardForwardEchoMethod.py`              | Echo hiding with backward and forward kernels                           | [[32]](#articles) |
| 23. | `TimeSpreadEchoMethod.py`                   | Time-spread echo with a pseudo-noise kernel                             | [[33]](#articles) |
| 24. | `HistogramMethod.py`                        | Histogram-based embedding robust to cropping and time-scale modification | [[34]](#articles) |
| 25. | `LowFrequencyAmplitudeMethod.py`            | Low-frequency amplitude modification (LFAM)                             | [[35]](#articles) |
| 26. | `AudioSealMethod.py`                        | AudioSeal neural watermarking (pretrained)                              | [[36]](#articles) |
| 27. | `WavMarkMethod.py`                          | WavMark neural watermarking (pretrained)                                | [[37]](#articles) |

All methods implement the abstract interface `SteganographyMethod`:

```python
from abc import abstractmethod, ABC
from typing import List
import numpy as np


class SteganographyMethod(ABC):

    @abstractmethod
    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        ...

    @abstractmethod
    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        ...

    @abstractmethod
    def type(self) -> str:
        ...
```

Packaged methods are registered in `taf/methods/factory.py` and in the `MethodType` enumeration. External methods are
registered as [plugins](#plugins). Both must satisfy the [method contract](#about).

<a id="metrics"></a>

## 6. Objective quality metrics

Metrics are computed between the cover `x` and the processed signal. They are grouped by the property they estimate
(Tables 2–5). Numbering is continuous across the tables.

Each metric declares its direction (`higher_is_better`), which rankings and significance tests use instead of inferring
it from the metric name. A metric that returns several numbers declares their names (`components`), and each is
reported separately rather than averaged. For example, BSSEval reports SDR, ISR, SIR and SAR, and a permutation index
that is never ranked. PESQ reports the raw P.862 score and MOS-LQO. CSII reports its high-, mid- and low-level indices.
For the non-intrusive SRMR and MOSNet, only the score of the processed signal is ranked, and the cover score is reported
as a reference. STGI and wSTMI, which are defined at 10 kHz, resample other rates first, as STOI does.

<a id="ai-based"></a>

#### 6.1 Data-driven (AI-based) metrics

**Table 2.** Learned quality predictors.

| No. | Module            | Metric                                                     | Ref.              |
|-----|-------------------|------------------------------------------------------------|-------------------|
| 1.  | `MosNetMetric.py` | MOSNet, deep-learning mean-opinion-score prediction        | [[16]](#articles) |

<a id="speech-reverberation"></a>

#### 6.2 Speech reverberation

**Table 3.** Reverberation- and spectral-distortion measures.

| No. | Module          | Metric                                                  | Ref.              |
|-----|-----------------|---------------------------------------------------------|-------------------|
| 2.  | `BsdMetric.py`  | Bark spectral distortion (BSD)                          | [[7]](#articles)  |
| 3.  | `SrmrMetric.py` | Speech-to-reverberation modulation energy ratio (SRMR)  | [[10]](#articles) |

<a id="speech-intelligibility"></a>

#### 6.3 Speech intelligibility

**Table 4.** Intelligibility predictors.

| No. | Module          | Metric                                             | Ref.             |
|-----|-----------------|----------------------------------------------------|------------------|
| 4.  | `CsiiMetric.py` | Coherence speech intelligibility index (CSII)      | [[7]](#articles) |
| 5.  | `NcmMetric.py`  | Normalised covariance measure (NCM)                | [[7]](#articles) |
| 6.  | `StoiMetric.py` | Short-time objective intelligibility (STOI)        | [[9]](#articles) |

<a id="speech-quality"></a>

#### 6.4 Speech quality

**Table 5.** Signal-fidelity and perceptual quality measures.

| No. | Module                         | Metric                                                   | Ref.              |
|-----|--------------------------------|----------------------------------------------------------|-------------------|
| 7.  | `SnrMetric.py`                 | Signal-to-noise ratio (SNR)                              | [[12]](#articles) |
| 8.  | `MelCepstralDistanceMetric.py` | Mel-cepstral distance (MCD)                              | [[11]](#articles) |
| 9.  | `SnrSegMetric.py`              | Segmental SNR (SNRseg)                                   | [[7]](#articles)  |
| 10. | `FWSnrSegMetric.py`            | Frequency-weighted segmental SNR (fwSNRseg)              | [[7]](#articles)  |
| 11. | `CepstrumDistanceMetric.py`    | Cepstral distance (CD)                                   | [[7]](#articles)  |
| 12. | `LlrMetric.py`                 | Log-likelihood ratio (LLR)                               | [[7]](#articles)  |
| 13. | `WssMetric.py`                 | Weighted spectral slope (WSS)                            | [[7]](#articles)  |
| 14. | `PesqMetric.py`                | Perceptual evaluation of speech quality (PESQ)           | [[8]](#articles)  |
| 15. | `CsigMetric.py`                | Composite signal-distortion rating (Csig)                | [[13]](#articles) |
| 16. | `CovlMetric.py`                | Composite overall-quality rating (Covl)                  | [[13]](#articles) |
| 17. | `CbakMetric.py`                | Composite background-intrusiveness rating (Cbak)         | [[13]](#articles) |
| 18. | `WstmiMetric.py`               | Weighted spectro-temporal modulation index (wSTMI)       | [[14]](#articles) |
| 19. | `StgiMetric.py`                | Spectro-temporal glimpsing index (STGI)                  | [[15]](#articles) |
| 20. | `SisdrMetric.py`               | Scale-invariant signal-to-distortion ratio (SI-SDR)      | [[17]](#articles) |
| 21. | `BSSEvalMetric.py`             | BSSEval v4 source-separation measures                    | [[18]](#articles) |

All metrics implement the abstract interface `Metric`:

```python
from abc import ABC, abstractmethod
from numbers import Number

import numpy as np


class Metric(ABC):

    @abstractmethod
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        ...

    @abstractmethod
    def name(self) -> str:
        ...
```

<a id="steganalysis"></a>

## 7. Steganalysis

Transparency metrics quantify the perceptual cost of embedding and attacks quantify the survival of the payload, but
neither establishes whether the *presence* of a payload can be inferred — the criterion that distinguishes
steganography from watermarking. `taf.steganalysis` estimates this empirically.

Cover signals are segmented into windows; each window is paired with its stego counterpart, and the pairs are split
into disjoint training and test sets so that no cover contributes to both. Residual-Markov and log-spectral features
are extracted, and an ensemble of Fisher linear discriminants trained on random feature subspaces
[[39]](#articles) is fitted to the training set. The procedure reports test accuracy, false-positive and
false-negative rates and the out-of-bag error of the ensemble.

```python
from taf.steganalysis import measure_detectability
from taf.methods.factory import SteganographyMethodFactory
from taf.models.types import MethodType

result = measure_detectability(
    SteganographyMethodFactory.get(16000, MethodType.LSB_METHOD),
    covers,                 # mono waveforms, segmented into windows internally
    message_length=20,
)
print(result.accuracy, result.false_positive_rate, result.undetectable)
```

A test accuracy of `0.5` corresponds to chance level and `1.0` to perfect detection. The result carries a 95% Wilson
interval of the accuracy and a one-sided exact binomial test against chance; `significantly_detectable` is `True` when
p < 0.05. The older `undetectable` flag (accuracy ≤ 0.55) is a heuristic on the point estimate, and with a small test
set it can disagree with the test. The result is a lower bound on detectability: a negative outcome does not exclude
detection by stronger features or classifiers. The `detectability` experiment type runs this analysis for every
selected method and payload length. Its window length and test fraction are set through `advanced_options`.

<a id="attacks"></a>

## 8. Attack and channel model

Attacks model the transformations a stego signal may undergo between embedding and extraction: incidental processing,
storage and transmission, format conversion, acoustic playback and re-recording, and intentional removal attempts. Each
attack is an immutable dataclass whose fields are its parameters. Applying it returns the processed signal together
with metadata sufficient to reproduce the trial exactly.

```python
from taf.attacks import build

result = build("awgn:snr_db=20,seed=7").apply(samples, sample_rate)
result.audio
result.metadata["parameters"]      # {"snr_db": 20.0, "seed": 7, ...}
result.metadata["measured_snr_db"]
```

The following design principles are enforced:

* **Reproducibility** — all stochastic attacks use an explicit seed; the global random state is never used. Within
  an experiment the seed is derived per trial (see [Experimental protocol](#platform)) unless the specification pins
  one.
* **Explicit parameters** — a severity label (e.g. `mp3@strong`) resolves to numeric parameters that are stored in
  every result row, so results are interpretable without consulting the source code.
* **Sampling-rate awareness** — filter cut-offs and resampling targets are derived from `f_s` and validated against
  the Nyquist frequency, so a suite is equally valid at 16 kHz and 44.1 kHz.
* **No self-normalisation** — an attack never rescales its output to compensate for its own effect.

**Table 6.** Attack families.

| Family | Attacks |
| --- | --- |
| Codec | `mp3`, `aac`, `opus`, `vorbis` — real FFmpeg encode/decode round trips |
| Noise | `awgn`, `pink_noise`, `impulse_noise` — parameterised by target SNR |
| Filtering | `low_pass`, `high_pass`, `band_pass`, `notch`, `smoothing` |
| Resampling | `resample` (round trip), `clock_drift` |
| Quantisation | `bit_depth` |
| Amplitude | `gain` (dB), `clipping`, `compression_dynamic` |
| Temporal | `time_shift`, `crop`, `zero_padding`, `sample_jitter`, `dropout`, `time_stretch`, `speed`, `pitch_shift` |
| Acoustic | `echo`, `reverb`, `acoustic_channel` |
| Pipelines | `streaming_upload`, `voice_call`, `broadcast`, `over_the_air`, `desync_attack` |

An attack specification may carry explicit parameters (`"awgn:snr_db=20"`), a severity level (`"mp3@strong"`) or refer
to a composite pipeline (`"pipeline:name=voice_call"`). Predefined benchmark suites cover all families at several
severity levels:

```yaml
experiment_type: attack_robustness
attack_preset: standard     # quick | standard | full
```

Codec attacks require FFmpeg on `PATH`. The parameter rationale, an audit of the previous implementation and known
limitations are documented in [docs/attacks.md](docs/attacks.md).

<a id="references"></a>

## 9. References

<a id="articles"></a>

#### 9.1 Literature

[1] A. A. Alsabhany, A. H. Ali, F. Ridzuan, A. H. Azni, and M. R. Mokhtar, "Digital Audio Steganography: Systematic Review, Classification, and Analysis of the Current State of the Art," *Computer Science Review*, vol. 38, article 100316, 2020. [doi:10.1016/j.cosrev.2020.100316](https://doi.org/10.1016/j.cosrev.2020.100316)

[2] H. T. Hu and L. Y. Hsu, "Robust, Transparent and High-Capacity Audio Watermarking in DCT Domain," *Signal Processing*, vol. 109, pp. 226-235, 2015. [doi:10.1016/j.sigpro.2014.11.011](https://doi.org/10.1016/j.sigpro.2014.11.011)

[3] I. Natgunanathan, Y. Xiang, G. Hua, G. Beliakov, and J. Yearwood, "Patchwork-Based Multilayer Audio Watermarking," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 25, no. 11, pp. 2176-2187, 2017. [doi:10.1109/TASLP.2017.2749001](https://doi.org/10.1109/TASLP.2017.2749001)

[4] S. Saadi, A. Merrad, and A. Benziane, "Novel Secured Scheme for Blind Audio/Speech Norm-Space Watermarking by Arnold Algorithm," *Signal Processing*, vol. 154, pp. 74-86, 2019. [doi:10.1016/j.sigpro.2018.08.011](https://doi.org/10.1016/j.sigpro.2018.08.011)

[5] J. Zhao, T. Zong, Y. Xiang, L. Gao, W. Zhou, and G. Beliakov, "Desynchronization Attacks Resilient Watermarking Method Based on Frequency Singular Value Coefficient Modification," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, pp. 2282-2295, 2021. [doi:10.1109/TASLP.2021.3092555](https://doi.org/10.1109/TASLP.2021.3092555)

[6] R. M. Nugraha, "Implementation of Direct Sequence Spread Spectrum Steganography on Audio Data," in *Proceedings of the 2011 International Conference on Electrical Engineering and Informatics (ICEEI)*, 2011. [doi:10.1109/ICEEI.2011.6021662](https://doi.org/10.1109/ICEEI.2011.6021662)

[7] P. C. Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed. CRC Press, 2013. [doi:10.1201/b14529](https://doi.org/10.1201/b14529)

[8] M. Wang, C. Boeddeker, R. G. Dantas, and A. Seelan, "PESQ (Perceptual Evaluation of Speech Quality) Wrapper for Python Users," Zenodo, 2022. [doi:10.5281/zenodo.6549559](https://doi.org/10.5281/zenodo.6549559)

[9] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, "A Short-Time Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech," in *Proceedings of ICASSP 2010*, Dallas, TX, USA, 2010. [doi:10.1109/ICASSP.2010.5495701](https://doi.org/10.1109/ICASSP.2010.5495701)

[10] T. H. Falk, C. Zheng, and W.-Y. Chan, "A Non-Intrusive Quality and Intelligibility Measure of Reverberant and Dereverberated Speech," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 18, no. 7, pp. 1766-1774, 2010. [doi:10.1109/TASL.2010.2052247](https://doi.org/10.1109/TASL.2010.2052247)

[11] R. Kubichek, "Mel-Cepstral Distance Measure for Objective Speech Quality Assessment," in *Proceedings of the IEEE Pacific Rim Conference on Communications, Computers and Signal Processing*, Victoria, BC, Canada, vol. 1, pp. 125-128, 1993. [doi:10.1109/PACRIM.1993.407206](https://doi.org/10.1109/PACRIM.1993.407206)

[12] Wikipedia contributors, "Signal-to-Noise Ratio," *Wikipedia*. [https://en.wikipedia.org/wiki/Signal-to-noise_ratio](https://en.wikipedia.org/wiki/Signal-to-noise_ratio)

[13] Y. Hu and P. C. Loizou, "Evaluation of Objective Quality Measures for Speech Enhancement," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 16, no. 1, pp. 229-238, 2008. [doi:10.1109/TASL.2007.911054](https://doi.org/10.1109/TASL.2007.911054)

[14] A. Edraki, W.-Y. Chan, J. Jensen, and D. Fogerty, "Speech Intelligibility Prediction Using Spectro-Temporal Modulation Analysis," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, pp. 210-225, 2021. [doi:10.1109/TASLP.2020.3039929](https://doi.org/10.1109/TASLP.2020.3039929)

[15] A. Edraki, W.-Y. Chan, J. Jensen, and D. Fogerty, "A Spectro-Temporal Glimpsing Index (STGI) for Speech Intelligibility Prediction," in *Proceedings of Interspeech 2021*, 2021. [doi:10.21437/Interspeech.2021-605](https://doi.org/10.21437/Interspeech.2021-605)

[16] C.-C. Lo, S.-W. Fu, W.-C. Huang, X. Wang, J. Yamagishi, Y. Tsao, and H.-M. Wang, "MOSNet: Deep Learning Based Objective Assessment for Voice Conversion," *arXiv preprint*, arXiv:1904.08352, 2019. [arXiv:1904.08352](https://arxiv.org/abs/1904.08352)

[17] J. Le Roux, S. Wisdom, H. Erdogan, and J. R. Hershey, "SDR - Half-Baked or Well Done?," in *Proceedings of ICASSP 2019*, 2019. [doi:10.1109/ICASSP.2019.8683855](https://doi.org/10.1109/ICASSP.2019.8683855)

[18] F.-R. Stöter, A. Liutkus, and N. Ito, "The 2018 Signal Separation Evaluation Campaign," in *Latent Variable Analysis and Signal Separation*, LVA/ICA 2018, pp. 293-305, 2018. [doi:10.5281/zenodo.3376621](https://doi.org/10.5281/zenodo.3376621)

[19] G. Yang, "An Improved Phase Coding Audio Steganography Algorithm," *arXiv preprint*, arXiv:2408.13277, 2024. [doi:10.48550/arXiv.2408.13277](https://doi.org/10.48550/arXiv.2408.13277)

[20] P. K. Dhar and T. Shimamura, "Blind SVD-Based Audio Watermarking Using Entropy and Log-Polar Transformation," *Journal of Information Security and Applications*, vol. 20, pp. 74-83, 2015. [doi:10.1016/j.jisa.2014.10.007](https://doi.org/10.1016/j.jisa.2014.10.007)

[21] F. A. Adhiyaksa, T. Ahmad, A. M. Shiddiqi, B. J. Santoso, H. Studiawan, and B. A. Pratomo, "Reversible Audio Steganography Using Least Prime Factor and Audio Interpolation," in *Proceedings of the 2021 International Seminar on Machine Learning, Optimization, and Data Science (ISMODE)*, pp. 97-102, 2022. [doi:10.1109/ISMODE53584.2022.9743066](https://doi.org/10.1109/ISMODE53584.2022.9743066)

[22] S. Mushtaq, S. Mehraj, and S. A. Parah, "Blind and Robust Watermarking Framework for Audio Signals," in *Proceedings of the 2024 11th International Conference on Reliability, Infocom Technologies and Optimization (ICRITO)*, pp. 1-5, 2024. [doi:10.1109/ICRITO61523.2024.10522195](https://doi.org/10.1109/ICRITO61523.2024.10522195)

[23] J. Wang and K. Wang, "A Novel Audio Steganography Based on the Segmentation of the Foreground and Background of Audio," *Computers & Electrical Engineering*, vol. 117, article 109247, 2025. [doi:10.1016/j.compeleceng.2024.109247](https://doi.org/10.1016/j.compeleceng.2024.109247)

[24] J. Yan, Y. Cheng, Z. Yin, X. Zhang, S. Wang, T. Sun, and X. Jiang, "FGAS: Fixed Decoder Network-Based Audio Steganography with Adversarial Perturbation Generation," *arXiv preprint*, arXiv:2505.22266, 2025. [arXiv:2505.22266](https://arxiv.org/abs/2505.22266)

[25] W. Luo, Y. Zhang, and H. Li, "Adaptive Audio Steganography Based on Advanced Audio Coding and Syndrome-Trellis Coding," in *Digital Forensics and Watermarking, IWDW 2017*, Lecture Notes in Computer Science, vol. 10431, pp. 177-186. Springer, 2017. [doi:10.1007/978-3-319-64185-0_14](https://doi.org/10.1007/978-3-319-64185-0_14)

[26] Y. Yan, Y. Li, Q. Xiao, and Y. Ren, "PRoADS: Provably Secure and Robust Audio Diffusion Steganography with Latent Optimization and Backward Euler Inversion," *arXiv preprint*, arXiv:2603.10314, 2026. [arXiv:2603.10314](https://arxiv.org/abs/2603.10314)

[27] A. A. Hamdi, A. A. Eyssa, M. I. Abdalla, M. ElAffendi, A. A. S. AlQahtani, A. A. Ateya, and R. A. Elsayed, "Improving Audio Steganography Transmission over Various Wireless Channels," *Journal of Sensor and Actuator Networks*, vol. 14, no. 6, article 106, 2025. [doi:10.3390/jsan14060106](https://doi.org/10.3390/jsan14060106)

[28] J. Nayeem, H.-B. Lee, and Y.-H. Seo, "Robust Audio Watermarking with Learnable Embedding Technique and Genetic Optimization," *Digital Signal Processing*, vol. 183, article 106372, 2026. [doi:10.1016/j.dsp.2026.106372](https://doi.org/10.1016/j.dsp.2026.106372)

[29] B. Chen and G. W. Wornell, "Quantization Index Modulation: A Class of Provably Good Methods for Digital Watermarking and Information Embedding," *IEEE Transactions on Information Theory*, vol. 47, no. 4, pp. 1423-1443, 2001. [doi:10.1109/18.923725](https://doi.org/10.1109/18.923725)

[30] H. S. Malvar and D. A. F. Florencio, "Improved Spread Spectrum: A New Modulation Technique for Robust Watermarking," *IEEE Transactions on Signal Processing*, vol. 51, no. 4, pp. 898-905, 2003. [doi:10.1109/TSP.2003.809385](https://doi.org/10.1109/TSP.2003.809385)

[31] I. J. Cox, J. Kilian, F. T. Leighton, and T. Shamoon, "Secure Spread Spectrum Watermarking for Multimedia," *IEEE Transactions on Image Processing*, vol. 6, no. 12, pp. 1673-1687, 1997. [doi:10.1109/83.650120](https://doi.org/10.1109/83.650120)

[32] H. J. Kim and Y. H. Choi, "A Novel Echo-Hiding Scheme with Backward and Forward Kernels," *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 13, no. 8, pp. 885-889, 2003. [doi:10.1109/TCSVT.2003.815950](https://doi.org/10.1109/TCSVT.2003.815950)

[33] B.-S. Ko, R. Nishimura, and Y. Suzuki, "Time-Spread Echo Method for Digital Audio Watermarking," *IEEE Transactions on Multimedia*, vol. 7, no. 2, pp. 212-221, 2005. [doi:10.1109/TMM.2005.843366](https://doi.org/10.1109/tmm.2005.843366)

[34] S. Xiang and J. Huang, "Histogram-Based Audio Watermarking Against Time-Scale Modification and Cropping Attacks," *IEEE Transactions on Multimedia*, vol. 9, no. 7, pp. 1357-1372, 2007. [doi:10.1109/TMM.2007.906580](https://doi.org/10.1109/TMM.2007.906580)

[35] W.-N. Lie and L.-C. Chang, "Robust and High-Quality Time-Domain Audio Watermarking Based on Low-Frequency Amplitude Modification," *IEEE Transactions on Multimedia*, vol. 8, no. 1, pp. 46-59, 2006. [doi:10.1109/TMM.2005.861292](https://doi.org/10.1109/TMM.2005.861292)

[36] R. San Roman, P. Fernandez, H. Elsahar, A. Defossez, T. Furon, and T. Tran, "Proactive Detection of Voice Cloning with Localized Watermarking," in *Proceedings of the 41st International Conference on Machine Learning (ICML)*, 2024. [arXiv:2401.17264](https://arxiv.org/abs/2401.17264)

[37] G. Chen, Y. Wu, S. Liu, T. Liu, X. Du, and F. Wei, "WavMark: Watermarking for Audio Generation," *arXiv preprint*, arXiv:2308.12770, 2023. [arXiv:2308.12770](https://arxiv.org/abs/2308.12770)

[38] H. Liu, M. Guo, Z. Jiang, L. Wang, and N. Z. Gong, "AudioMarkBench: Benchmarking Robustness of Audio Watermarking," in *Advances in Neural Information Processing Systems 37 (NeurIPS Datasets and Benchmarks)*, 2024. [arXiv:2406.06979](https://arxiv.org/abs/2406.06979)

[39] J. Kodovsky, J. Fridrich, and V. Holub, "Ensemble Classifiers for Steganalysis of Digital Media," *IEEE Transactions on Information Forensics and Security*, vol. 7, no. 2, pp. 432-444, 2012. [doi:10.1109/TIFS.2011.2175919](https://doi.org/10.1109/TIFS.2011.2175919)

[40] T. Filler, J. Judas, and J. Fridrich, "Minimizing Additive Distortion in Steganography Using Syndrome-Trellis Codes," *IEEE Transactions on Information Forensics and Security*, vol. 6, no. 3, pp. 920-935, 2011. [doi:10.1109/TIFS.2011.2134094](https://doi.org/10.1109/TIFS.2011.2134094)

<a id="links"></a>

#### 9.2 Software resources

The following open-source projects served as references for, or are wrapped by, individual components.

| Ref. | Project | Scope |
| --- | --- | --- |
| [S1] | [audio-watermarking](https://github.com/kosta-pmf/audio-watermarking) | Audio watermarking and steganography implementations |
| [S2] | [audio-steganography-algorithms](https://github.com/ktekeli/audio-steganography-algorithms) | Audio steganography algorithm examples |
| [S3] | [pysepm](https://github.com/schmiph2/pysepm) | Objective speech enhancement and quality measures |
| [S4] | [PESQ](https://github.com/ludlows/PESQ) | Python wrapper for ITU-T P.862 PESQ |
| [S5] | [pystoi](https://github.com/mpariente/pystoi) | STOI implementation |
| [S6] | [SRMRpy](https://github.com/jfsantos/SRMRpy) | SRMR implementation |
| [S7] | [mel_cepstral_distance](https://github.com/jasminsternkopf/mel_cepstral_distance) | Mel-cepstral distance implementation |
| [S8] | [semetrics](https://github.com/nglehuy/semetrics) | Speech enhancement measures |
| [S9] | [py-intelligibility](https://github.com/aminEdraki/py-intelligibility) | wSTMI and STGI intelligibility measures |
| [S10] | [speechmetrics](https://github.com/aliutkus/speechmetrics) | Speech and audio evaluation measures |
| [S11] | [sigsep-mus-eval](https://github.com/sigsep/sigsep-mus-eval) | BSSEval v4 |

<a id="licence"></a>

## 10. Licence

The A-Files is free software distributed under the GNU General Public License, version 3 (GPLv3).

<a id="dependencies"></a>

## 11. External dependencies

* **PESQ** requires Microsoft Visual C++ 14.0 or later, available through the
  [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
* **FFmpeg** must be available on `PATH` for codec attacks and some format conversions: <https://ffmpeg.org/>.

<a id="authors"></a>

## 12. Authors

- Paweł Kaczmarek ([@pawel-kaczmarek](https://github.com/pawel-kaczmarek)) — Military University of Technology,
  Faculty of Electronics
- Zbigniew Piotrowski — Military University of Technology, Faculty of Electronics
