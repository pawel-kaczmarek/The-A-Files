# The A-Files: shared agent steering

## Scope, roles and skills

These instructions apply throughout this repository to Codex and other agents. Claude imports them through `CLAUDE.md`. Read applicable nested `AGENTS.md` files before editing a subtree, even if your client does not load them automatically. Directory instructions extend this shared guidance.

Work with the rigor of an audio signal-processing researcher and a senior software engineer/architect. These are working roles, not claims of human credentials. Prioritize scientific validity, reproducibility, maintainability and usable research tools. Keep changes proportional to the task.

Read the relevant project skill before specialized work; combine them for tasks crossing domains:

- Research, DSP, experiments and scientific writing: `.agents/skills/taf-audio-research/SKILL.md`.
- Python, backend architecture, APIs and persistence: `.agents/skills/taf-python-engineering/SKILL.md`.
- Frontend architecture, UI, accessibility and visualization: `.agents/skills/taf-research-ui/SKILL.md`.

Canonical skills live in `.agents/skills/`. Claude discovery entries in `.claude/skills/` reference them to avoid duplicating instructions.

## Scientific standards

Separate published claims, implementation assumptions and local measurements. Never invent citations, results or successful checks. Document deviations from papers and approximation limits. Keep imperceptibility, robustness, capacity and detectability distinct; quality scores do not establish security.

Preserve seeds, configuration and data provenance. Use audio files as the unit of replication in the existing analysis pipeline rather than repeated rows from one file. Preserve failed trials with missing scores and explicit failure reasons. Respect declared metric directions and components. Keep scientific calculations and aggregation in Python; the web client presents API results.

## Project Structure & Module Organization
The package lives in `src/taf/`; the historical `TAF/` layout is obsolete. Add algorithms under `src/taf/methods/`, contracts under `models/`, audio I/O under `audio/`, attacks under `attacks/`, and metrics under `metrics/`. Execution and study design live in `evaluation/` and `experiments/`; FastAPI and database code live in `api/` and `persistence/`. Keep packaged audio under `src/taf/resources/`, documentation and diagrams under `docs/`, and the Next.js research client under `web/`. Register new methods and metrics in the relevant `factory.py` and update types/catalogues as required.

## Build, Test, and Development Commands
`python -m venv venv` creates a local virtual environment.

`.\venv\Scripts\Activate.ps1` activates it on PowerShell.

`python -m pip install -e .[dev]` installs the package plus development dependencies from `pyproject.toml`.

`python -m pytest tests/` runs the Python suite. Prefer focused tests for affected behavior. The historical `scripts/smoke_workflow.py` is absent; use current tests and documented CLI workflows.

`python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('src/taf').rglob('*.py')]"` performs syntax-only validation without writing `__pycache__` files.

## Coding Style & Naming Conventions
Use 4-space indentation and standard Python naming: `snake_case` for functions and variables, `PascalCase` for classes, and explicit module names such as `LwtMethod.py` or `PesqMetric.py`. Preserve type hints on abstract interfaces like `SteganographyMethod` and `Metric`. Follow the existing factory-based registration pattern instead of adding ad hoc imports or branching logic.

## Testing Guidelines
Tests already live in `tests/`. Add focused regression or contract tests for changed behavior. For audio changes, use representative bundled VCTK and LibriSpeech samples when relevant and inspect decode success and metric output. Report checks actually run, skipped dependencies and material limitations. Backend integration checks may require PostgreSQL and `.[dev,platform]`; frontend validation is described in `web/AGENTS.md`.

## Commit & Pull Request Guidelines
Recent commits follow `The A-Files - <type>: <subject>`; examples include `add`, `fix`, and `update`. Keep messages imperative and scoped to one algorithm, metric, or dependency change. Pull requests should describe the affected module, note any requirement changes, list validation steps, and include representative metric or decode results when behavior changes.

## Dependencies & Configuration Tips
The README notes two external prerequisites: `pesq` may require Microsoft Visual C++ Build Tools, and some workflows may need FFmpeg installed. Avoid committing large generated assets or extra datasets unless they are required for reproducible evaluation.
