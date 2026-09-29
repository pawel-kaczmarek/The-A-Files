# CLAUDE.md

@AGENTS.md

The imported file defines shared scientific and engineering standards. Read applicable nested `AGENTS.md` files before editing a subtree. Project skills in `.claude/skills/` reference canonical instructions in `.agents/skills/`. The architecture notes below supplement these rules; verify changeable details against source code.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The A-Files is an audio steganography research toolkit for embedding secret data in audio signals, with robustness testing and signal quality metrics. The package name is `taf`, version `2.0.0`, licensed under GPLv3.

## Commands

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install package with dev dependencies
python -m pip install -e .[dev]

# Run tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_package_import.py

# Syntax-only validation (no __pycache__)
python -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('src/taf').rglob('*.py')]"
```

**External prerequisites:**
- `pesq` may require Microsoft Visual C++ Build Tools
- Some workflows may need FFmpeg installed

**Research platform** (PostgreSQL + API + web client):

```powershell
docker compose up -d db                 # PostgreSQL 17, taf/taf/taf on localhost:5432
python -m pip install -e .[dev,platform]
taf-api                                 # http://127.0.0.1:8000, applies Alembic migrations on start
cd web; npm install; npm run dev        # http://localhost:3000
```

Containerised platform: `docker compose up -d --build` builds `Dockerfile` (API; `TAF_EXTRAS` build arg, FFmpeg,
`TAF_DATA_DIR=/data` volume) and `web/Dockerfile` (Next.js standalone via `TAF_NEXT_OUTPUT=standalone`;
`NEXT_PUBLIC_TAF_API_URL` is inlined at build time from `TAF_PUBLIC_API_URL`). On this machine Docker runs in WSL:
`wsl.exe -e docker ...`.

`tests/test_api.py` uses `TAF_TEST_DATABASE_URL` (default database `taf_test` on the same server) and is skipped when
PostgreSQL is unreachable.

## Architecture

Package source lives under `src/taf/` (modern src-layout). Key modules:

- **`models/`** — Abstract base classes: `SteganographyMethod` (`encode`, `decode`, `type`), `Metric` (`calculate`, `name`), `WavFile` dataclass, and `types.py` enums (`MethodType`, `MetricType`).
- **`methods/`** — 30 steganography algorithm implementations, all extending `SteganographyMethod`. New algorithms go here and must be registered in `BUILTIN_METHODS` (`methods/factory.py`) *and* in `MethodType` (`models/types.py`). Over-capacity messages raise `CapacityError` (`models/errors.py`), not a bare `ValueError`. Two of them (`AudioSealMethod`, `WavMarkMethod`) wrap pretrained neural models from the optional `[neural]` extra and import lazily.
- **`metrics/`** — 21 metric implementations grouped into `speech_quality/`, `speech_intelligibility/`, `speech_reverberation/`, and `ai_based/mosnet/`. New metrics must be registered in `BUILTIN_METRICS` (`metrics/factory.py`) and must declare `higher_is_better`; a metric returning several numbers must also declare `components` (and `component_directions` for entries that are not scores) so the engine reports each entry instead of averaging them.
- **`attacks/`** — Attack package organised by phenomenon: `base.py` (Attack/AttackResult, dtype and channel handling, Nyquist validation), `noise.py`, `codec.py`, `filtering.py`, `resampling.py`, `quantization.py`, `amplitude.py`, `temporal.py`, `acoustic.py`, `pipeline.py`, `presets.py` (severity levels and benchmark suites), `registry.py` (builds attacks from specification strings), and `attacks.py` (the chainable `CorruptedWavFile` facade). Every attack is a frozen dataclass whose fields are its parameters; `apply(audio, sample_rate)` returns audio plus reproducibility metadata. New attacks must be added to `ATTACK_CLASSES` in `registry.py` and given a severity mapping in `presets.py`.
- **`audio/`** — `load_audio()` / `save_audio()` supporting WAV, FLAC, OGG; `AudioFileFormat` enum.
- **`evaluation/workflow.py`** — Async evaluation engine. Core types: `EvaluationConfig`, `EvaluationMessage`, `EvaluationRow`, `EvaluationResult`, `FailureKind`. Supports parallel execution via asyncio with semaphore-controlled concurrency. Decodes with a fresh method instance, computes cover-vs-stego metrics once per encoded signal, and derives per-trial seeds for messages and attacks in `evaluation/seeding.py`.
- **`experiments/`** — Experiment layer over the engine: `schema.py` (`ExperimentConfig`, nine `ExperimentType`s, `attack_sweep`/`method_sweep`), `sweeps.py` (`ParameterSweep`, sweep expansion, `threshold_crossing`), `runner.py` (dataset resolvers; `register_dataset_resolver` lets the platform add `library:<id>`), `results.py` (normalized rows; failed trials have `ber=None` and a `failure_kind`), `analysis.py` (cluster bootstrap by file, paired Wilcoxon/Friedman tests with Holm correction, box summaries, paired differences, file-stratified Spearman permutation test, Pareto front), `scenarios/` (per-type validation, summaries and design metadata: property, question, factors, measures, analyses; includes `robustness_curve.py`, `tradeoff_curve.py`, and `evaluation.py`, the evaluation block every factorial run gets in `summary["evaluation"]`: per-method BER distributions and recovery rates, per-attack BER increase, severity trends, quality vs attack damage, payload, runtime, correlations and data-derived findings), `registry.py` (catalogue listings consumed by the API), `inspector.py` (exact re-synthesis of a stored trial from its seed), `reporting.py` (LaTeX/Markdown run reports), `provenance.py` (run manifest), `detectability.py`, `uploads.py`.
- **`corpora/`** — `catalog.py` (standard corpora with licence, citation, DOI, download and speaker layout), `prepare.py` (reproducible subset preparation: seeded speaker-balanced selection, mono, polyphase resampling, 16-bit FLAC, SHA-256 manifest), `synthetic.py` (synthetic test signals).
- **`persistence/`** — PostgreSQL through SQLAlchemy 2: `models.py` (`Experiment` with versioned protocol, `Run` with configuration snapshot/summary/manifest, `ResultRowRecord` with indexed columns plus JSONB, `Dataset`), `store.py` (CRUD and row queries; `json_safe` turns NaN/Inf into null because JSONB rejects them), `session.py`, `settings.py` (`TAF_DATABASE_URL`, `TAF_DATA_DIR`), `migrations/` (Alembic, packaged). Schema changes need a new migration in `migrations/versions/`.
- **`api/`** — FastAPI: `app.py` (lifespan runs migrations), `runs.py` (`RunManager`: background execution, buffered row writes, SSE), `library.py` (dataset library and corpus preparation), `routers/` (`catalog`, `experiments`, `runs`, `datasets`), `schemas.py`. The web client in `web/` is a thin client: domain logic, aggregation, statistics and exports belong in Python.
- **`plugins.py`** — Catalogue of methods and metrics: packaged ones plus third-party ones from the `taf.methods` / `taf.metrics` / `taf.attacks` entry-point groups. Packaged names always win. Also parses method specifications (`"QIM_METHOD:step_scale=0.1"`, values are Python literals) and reports constructor parameters; `methods/catalog.py` adds family, purpose, reference, strength parameter and short name (`method_abbreviation`) for packaged methods.
- **`resources/`** — Packaged assets: `audio/example.wav`, `datasets/VCTK/16/` (10 FLAC files), `datasets/LibriSpeech/142345/` (11 FLAC files), `models/mosnet/cnn_blstm.h5`.
- **`steganalysis/`** — `measure_detectability()` trains an `EnsembleClassifier` (bagged Fisher discriminants on random feature subspaces) on residual-Markov and log-spectral features, and reports whether a method's output can be told apart from covers.
- **`generator/`** — Test payload helpers: `generate_sinus_waveform()`, `generate_noise()`, `generate_random_message()`.
- **`main.py`** — Entry point orchestrating the full evaluation workflow.

### Key patterns

- **Factory registration**: `SteganographyMethodFactory` and `MetricFactory` provide registry-based instantiation and build only the requested component. Always register new packaged methods/metrics in the relevant `factory.py` rather than adding ad hoc imports; external ones register through entry points (`taf.plugins`). Experiment configs name components by catalogue name, not by enum.
- **Layering**: `models`, `methods`, `metrics`, `attacks`, `audio`, `steganalysis`, `generator` and `corpora` never import `evaluation`, `experiments`, `persistence` or `api`; `evaluation` never imports `experiments`, `persistence` or `api`; `experiments` never imports `persistence` or `api`; `persistence` never imports `api`. Enforced by `tests/test_plugins.py`.
- **Statistics**: the file is the unit of replication. Never compute confidence intervals or tests treating rows of one file as independent, never score a failed trial as BER 1.0, and never rank metrics by a direction guessed from their name.
- **Evaluation block**: cross-design analyses belong in `scenarios/evaluation.py`, built only from `analysis.py` estimators; its comparisons join `summary["statistics"]` via `setdefault` rather than forming a second statistics tree. Bump `EVALUATION_VERSION` when its content changes: the API recomputes older stored summaries from their rows on first read. Findings must be computed statements with the figure they rest on, never free-text interpretation.
- **Abstract interfaces**: Preserve type hints on `SteganographyMethod` and `Metric` base classes.
- **Chainable attacks**: `CorruptedWavFile` implements a fluent builder pattern.
- **Method contract**: every method in `MethodType` is held to four properties by `tests/test_methods_roundtrip.py`, on real speech from the packaged VCTK subset: the message returns bit for bit through a *fresh* decoder instance; `encode()` neither writes into the caller's cover nor changes its length; an over-capacity message raises `CapacityError` instead of being silently truncated (also for more than four bits per sample); and encode/decode do not crash on synthetic audio. New methods must satisfy all four.
- **Attacks are parameterised, seeded and recorded**: never add an attack that draws from the global RNG, hard-codes a frequency without checking it against Nyquist, or normalises its own effect away. Severity labels must resolve to explicit numbers that reach the result row.
- **Relative, not absolute, strengths**: embedding strengths, quantisation steps and thresholds are expressed relative to a signal quantity the decoder can recompute (frame norm, band RMS, mean amplitude). Absolute constants break on quiet material and after any volume change; several methods were fixed for exactly this.

## Coding Conventions

- 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Explicit module names: `LwtMethod.py`, `PesqMetric.py`, etc.
- Commit messages: `The A-Files - <type>: <subject>` (e.g., `add`, `fix`, `update`) — imperative, scoped to one algorithm/metric/dependency.
- PRs should describe the affected module, note requirement changes, list validation steps, and include representative metric or decode results when behavior changes.
- Do not commit large generated assets or extra datasets unless required for reproducible evaluation.
