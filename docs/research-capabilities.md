# Research capabilities and protocol

This guide defines payloads, signal measurements, input provenance and literature comparisons.
The implementation lives in `src/taf`, with a FastAPI/PostgreSQL backend and a Next.js frontend in `web`.
For the execution model and inferential statistics, see [experiments](experiments.md).

## Measures and their meaning

| Registry name or result field | Definition and scope |
| --- | --- |
| `ESTOI_METRIC` | Extended STOI via the existing `pystoi` dependency, `extended=True`; higher indicates greater predicted **speech intelligibility**. Silent references and insufficient active frames are errors. Not a music-quality or security score. |
| `LSD_METRIC` | Mean over frames of RMS over bins of `20 log10(max(abs(STFT(x)), 1e-5)) − 20 log10(max(abs(STFT(y)), 1e-5))`, in dB. 32 ms Hann windows, 75% overlap, no centering or final-frame padding; includes DC/Nyquist. Fixed −100 dBFS magnitude floor. Lower is closer. |
| `MRSC_METRIC` | Mean of `||abs(STFT(x)) − abs(STFT(y))||F / ||abs(STFT(x))||F` at 16, 32 and 64 ms, Hann and 75% overlap. Silent reference is undefined. Lower is closer. |
| `VISQOL_METRIC` | Official Google ViSQOL **audio mode**, using its bundled `libsvm_nu_svr_model.txt`. Both metric inputs are polyphase-resampled to 48 kHz; the decoder's audio is unchanged. Returns MOS-LQO; higher is better. Optional bindings/model must be installed separately. |
| `payload_length`, `payload_bytes` | Exact submitted bits, and bit count / 8 (a byte-equivalent, potentially fractional). |
| `payload_rate_bps` | Offered message bits / cover duration. This is the actual load, including a request that exceeds a method's capacity; it is not proof of successful embedding. |
| `payload_bits_per_sample` | Submitted bits / (evaluated frames × evaluated channels); mono experiments use frames as the denominator. |
| `exact_goodput_bps` | Submitted bits / cover duration **only if the entire decoded message is exact**; zero for bit errors or failed delivery. No ECC or partial-byte recovery is assumed. This is observed delivered payload rate, not Shannon capacity. |
| `encode_rtf`, `decode_rtf` | Timed encode/decode call seconds / cover duration. Does not include loading, model construction outside the timed call, metrics or total application overhead. |

LSD and MRSC are complementary diagnostics, not perceptual MOS predictors. Their frame lengths are fixed
independently of the legacy interface's `frame_len`/`overlap` arguments. For clips shorter than a window, that
window is reduced to the clip length. They require equal-length finite mono arrays, and do not fit delay, gain
or phase. Magnitude-only diagnostics cannot detect every phase distortion; retain waveform/perceptual measures.

The eSTOI definition is [Jensen and Taal (2016)](https://doi.org/10.1109/TASLP.2016.2585878).
ViSQOL installation, model layout, input guidance and audio/speech mode distinctions follow the
[official implementation](https://github.com/google/visqol#python-api-usage). Install that source package using
its build instructions; no unrelated similarly named PyPI package is required by TAF. Its model hash and package
version are included in the run manifest. Missing bindings or model files become a preview warning and metric
error, never a substitute MOS. Prefer active clips around 8–10 seconds. Upsampling narrowband audio does not
make it full-band; compare matched native bandwidths, and do not mix audio-mode with speech-mode scores.

Imperceptibility uses `metrics` (cover ↔ stego). Attack damage uses `attack_metrics` (stego ↔ attacked).
Payload robustness uses BER and complete-message recovery under each attack condition. Attack quality is
retained even if decoding fails; sample-aligned metrics are omitted when an attack changes length, channels
or sampling rate. Decoder failure still has no BER. Extra decoded bits can leave positional BER at zero but
invalidate exact recovery; `decoded_length` makes this visible.

The experiment API retains its existing in-memory clean baseline (`DecodeTarget.DIRECT`). Successful clean
decoding therefore establishes recovery from that floating-point stego signal. Test PCM quantization or codec
roundtrips explicitly before claiming recovery from an exported audio file; floating-point LSB changes can
disappear on conversion to integer PCM.

The existing capacity scenario remains the empirical largest passing tested payload before the first failure,
with its declared BER/completion thresholds and censoring. Requested rate sweeps also work in this scenario.
A reported maximum is conditional on the tested material, method parameters, thresholds and tested grid.
If rate sweeps produce different bit-length grids across files, pooled maximum-bit claims are omitted;
per-file capacity and its bits/s summary remain available.

No KL divergence of amplitude histograms is added as a security score: it would not establish resistance to
steganalysis. No corpus-level Fréchet score is used as a paired clip metric. Peak process memory is deferred
because shared concurrent workloads would make an apparent per-trial number misleading. RTF comparisons in
the research view are disabled unless `max_workers=1`; single-worker wall timings still need matched hardware,
warm-up and controlled system load for strong performance claims.

## Payloads

Existing `payload_lengths=[16, 32]` behavior and length-derived random seeds are preserved (4–8192 bits).
The engine's generator is reused, so adding another requested length does not change existing messages.

```python
from taf.experiments import ExperimentConfig, run_experiment

config = ExperimentConfig(
    name="Matched offered rates", experiment_type="dataset_benchmark",
    dataset_id="vctk", methods=["LSB_METHOD", "QIM_METHOD"],
    metrics=["SNR_METRIC", "ESTOI_METRIC", "LSD_METRIC", "MRSC_METRIC"],
    payload_rates_bps=[8, 16, 32], repetitions=3, random_seed=42,
    subset_seed=7, file_limit=5, max_workers=1,
)
run = run_experiment(config)
run.to_csv("results.csv")
```

When rates are supplied, they replace the fixed-length list. Each file receives `floor(rate × frames / fs)`
bits; requests resolving outside 1–8192 bits are rejected, not clamped. `requested_payload_rate_bps` and the
actual rate are both retained. Random messages of the same resolved length/repetition share bits across
methods/files, consistent with the existing paired protocol. Different rates may round to the same length.

For exact content, use one of these JSON-compatible `payload` values:

```json
{"kind": "text", "value": "Research message"}
{"kind": "binary", "value": "0001ff"}
{"kind": "bits", "value": "00101"}
```

Text is UTF-8; text and binary bytes use MSB-first bit order. Binary is hex to preserve NULs and leading zero
bytes in JSON. Explicit bit strings may be non-byte-aligned. Explicit payloads contain 1–8192 bits and ignore
the random-length list; rate sweeps are disallowed for them. They are repeated unchanged, not described as
independent messages. Methods retain their own capacity/alignment constraints and report failures normally.
No implicit framing, padding, compression, encryption or ECC is added. The editor can read a binary file
(maximum 1024 bytes). Content is stored in exported configurations and message bits in result rows.
`payload_sha256` hashes the ASCII bit string, so non-byte-aligned messages have unambiguous identities.
The specialized detectability design continues to accept only fixed-length random payloads.

## Audio metadata, subsets and replay

Rows and run manifests now retain source channel count, PCM bit depth, subtype, source/category, relative file
identity, sample count, file hash and preprocessing. Duration is calculated from actual frames/sample rate.
Unknown categories stay `unknown`; integer PCM bit depth stays null for float/lossy subtypes. Loaded floating-point
array dtype is not presented as source bit depth. Evaluated channel count is separate from source channels.

`channel_policy="mono"` explicitly averages source channels before embedding and records that transformation.
`channel_policy="reject"` requires mono files. These are mono-method experiments, not stereo fidelity tests.
Prepared corpus manifests retain native header metadata and native member hashes alongside the processed
PCM_16 mono FLAC metadata. Existing resampling/excerpt/peak-scaling behavior is unchanged and documented.

`subset_seed` selects from stable relative-path ordering independently of payload/attack randomness, before
`file_limit`. `selected_files` accepts exact relative paths; a basename is accepted only when unambiguous.
Exported run configurations pin the actual selected paths and SHA-256 values in `selected_file_sha256`.
Adding unrelated files does not change replay; changing selected content fails a checksum check. The original
study configuration remains available for a new sampling exercise. Trial resynthesis uses the same preprocessing
and checks the recorded file hash. Statistical clusters use relative identity/path, not basename alone.

Mixed datasets can provide `manifest.json` with per-file `category`, `source` and `speaker`; measured header
fields are authoritative. For example:

```json
{"files": [
  {"file": "speech/clip.wav", "category": "speech", "source": "Study A", "speaker": "s01"},
  {"file": "music/clip.wav", "category": "music", "source": "Study B"}
]}
```

Optional manifest `sha256` values are checked. Local-library scans preserve annotations and add measured header
fields/digests. Library corpus domains propagate automatically. `audio_category`/`audio_source` override metadata
for a whole run, so leave them unset when preserving per-file categories in mixed datasets.

## Literature and UI

The version-controlled evidence catalog is exposed at `/api/catalog/literature` and `/literature`.
It supplements existing implemented-method citations with four reference-only studies:

- [Hide and Speak — Kreuk et al., Interspeech 2020](https://doi.org/10.21437/Interspeech.2020-2380): speech-in-speech steganography.
- [DeAR — Liu et al., AAAI 2023](https://doi.org/10.1609/aaai.v37i11.26550): learned watermarking with physical re-recording experiments.
- [SilentCipher — Singh et al., Interspeech 2024](https://doi.org/10.21437/Interspeech.2024-174): neural audio watermarking with perceptual constraints.
- [IDEAW — Li et al., EMNLP 2024](https://doi.org/10.18653/v1/2024.emnlp-main.258): invertible dual embedding with a separate locating code.

Each entry records authors, publication year/venue, DOI/arXiv, family/purpose, datasets, metrics, attacks,
payload definition, source URL and verification date. Numeric observations have table/section locators and
experimental conditions. They are reported by the papers, not reproduced locally. No new embedding algorithm
is claimed. In particular, reconstructed speech is not binary capacity, and synchronization bits are not useful
message bits. The UI searches all these fields, filters year/family/purpose and compares selected protocols.

The Statistics tab's Research comparisons section calls `/api/runs/{id}/research`. It filters material,
source, payload size/kind and requested rate, groups results, and draws existing interval charts with
file-bootstrap 95% CIs. Every view chooses
one attack condition and one quality reference. BER excludes failures while goodput includes failed deliveries
as zero; counts and finite metric coverage are shown. These are descriptive row-weighted comparisons;
unequal payload/audio mixtures are not evidence of a causal method advantage. The existing paired statistical
views remain available. Dataset and trial pages expose new metadata and exact payload identities.

CSV includes the new fields, method parameters and explicit JSON metric error fields. Old result rows remain
readable with missing fields shown as unknown. No database migration, model download or additional audio corpus
is needed for the core additions. New research editor/view labels and paper evidence are currently English;
existing localized views and navigation remain available.

## Validation and changed areas

Focused tests are in `tests/test_research_{metrics,payloads,audio,comparisons}.py`: known spectral identities/gain,
invalid/silent inputs, eSTOI reference parity, optional ViSQOL adapter contract, rate rounding and reproducibility,
UTF-8/NUL/bit handling, metadata, subset replay, digest drift, corpus preparation, failed-decoder quality retention,
file clustering, reference separation and HTTP catalog/comparison behavior. ViSQOL's actual numerical backend
test is skipped by the existing metric suite when the official dependency is absent; an adapter mock does not
validate perceptual accuracy. Run `python scripts/smoke_research.py` for bundled VCTK and LibriSpeech validation.

Validation commands (activate the project's virtual environment first):

```powershell
python -m pytest -q tests/test_research_metrics.py tests/test_research_payloads.py tests/test_research_audio.py tests/test_research_comparisons.py
$env:PGCONNECT_TIMEOUT = '3'
python -m pytest -q --ignore=tests/test_methods_roundtrip.py --ignore=tests/test_fgas_method.py
python scripts/smoke_research.py
cd web
npx tsc --noEmit
$env:TAF_NEXT_DIST_DIR = '.next-research-build'
npm run build
```

The regression command excludes the unchanged method-roundtrip and FGAS test modules. Earlier broader runs
were interrupted before completion; they are not recorded as full-suite passes.
Database integration tests require PostgreSQL; the new HTTP catalog/filter contract test uses an in-memory
FastAPI test application with a substituted store. It does not validate PostgreSQL persistence.

Validation recorded on 2026-09-27:

- Final focused research suite: **47 passed**.
- Regression run excluding the two modules above: **308 passed, 10 skipped** (nine PostgreSQL tests and
  the unavailable official ViSQOL numerical backend). Final capacity/statistics checks after the pooled-grid
  correction: **23 passed**. These counts overlap; they are not counts of distinct tests.
- Bundled VCTK and LibriSpeech smoke workflow: **32 trials**, with **16/16 exact clean decodes**; attacked
  BER and exact goodput were recorded separately. These are in-memory baseline results.
- Next.js production build in the isolated output directory and the final TypeScript check passed.
- Python syntax validation and `git diff HEAD --check` passed.

| Area | Files |
| --- | --- |
| Metrics | `src/taf/metrics/{catalog,factory}.py`, `common/spectral.py`, `speech_intelligibility/EstoiMetric.py`, `speech_quality/{LogSpectralDistanceMetric,SpectralConvergenceMetric,VisqolMetric}.py`, `src/taf/models/types.py` |
| Audio | `src/taf/audio/metadata.py`, `src/taf/models/WavFile.py`, `src/taf/corpora/{prepare,synthetic}.py`, `src/taf/api/library.py`, `src/taf/experiments/audio_inputs.py` |
| Payloads/engine | `src/taf/experiments/{payloads,schema,runner}.py`, `src/taf/evaluation/{config,messages,result,workflow,yaml_loader}.py` |
| Results/provenance | `src/taf/experiments/{results,research,analysis,provenance,inspector,csv_export,reporting,registry}.py`, `scenarios/embedding_capacity.py` |
| Literature/API | `src/taf/methods/literature.py`, `src/taf/api/routers/{catalog,runs}.py`, `src/taf/api/schemas.py` |
| Web | Literature page; metric, dataset, run and trial pages; `editor/{AudioOptions,PayloadEditor,ExperimentEditor,draft}`; `run/ResearchView`; shell, API/types/research helpers and navigation translations |
| Validation/docs | Four research test modules, `scripts/smoke_research.py`, README and this protocol; configurable isolated Next.js build output and ignore rules |
