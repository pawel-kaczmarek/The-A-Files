# The A-Files — web research platform

Next.js + TypeScript + Tailwind (shadcn/ui-style components) experiment
dashboard for the [The A-Files](../README.md) audio steganography toolkit. It
talks to the FastAPI layer shipped in the Python package; **all research logic
(encoding, decoding, attacks, metrics, aggregation, CSV generation) runs in
the backend** — the frontend only collects configuration, shows progress and
renders results.

> This directory is **not** part of the PyPI distribution — it is developed
> and deployed separately from the `taf` package.

## Prerequisites

- Node.js ≥ 18.18
- The taf API running locally:

```powershell
pip install -e ".[experiments]"
uvicorn taf.api.main:app --reload   # or: taf-api  → http://127.0.0.1:8000
```

## Development

```powershell
cd web
npm install
npm run dev   # http://localhost:3000
```

Set `NEXT_PUBLIC_TAF_API_URL` (see `.env.example`) if the API runs elsewhere.

## Routes

| Route | Purpose |
|---|---|
| `/dashboard` | Inventory counts, quick links, recent runs |
| `/experiments/dataset-benchmark` | General benchmark over a dataset |
| `/experiments/attack-robustness` | Method × attack robustness matrix |
| `/experiments/perceptual-quality` | Quality degradation per method/payload |
| `/experiments/embedding-capacity` | Payload sweep against pass/fail thresholds |
| `/experiments/method-comparison` | Weighted overall method ranking |
| `/experiments/research-experiment` | Fully configurable runner + config preview |
| `/experiments/[id]` | Live/completed run detail (summary, tables, exports) |
| `/results/history` | All runs of this API session (live via SSE) |
| `/results/exports` | Detailed/summary CSV + config JSON downloads |
| `/settings/{methods,metrics,attacks,datasets}` | Inventory pages incl. attack parameters and dataset upload |

## Architecture

- Every experiment page is a thin wrapper around the shared
  `components/experiments/ExperimentLayout.tsx` (config cards → preview → run
  → live progress → summary cards/tables → results table → CSV export).
- `lib/types.ts` mirrors the backend Pydantic models; `lib/validators.ts` only
  pre-checks what the backend re-validates authoritatively.
- Runs stream over Server-Sent Events (`/api/experiments/{id}/events`); the
  history list uses the global stream (`/api/experiments/events`). No polling.
- Results live in the API's memory for the session — use the CSV exports to
  persist them.

## Known limitations

- Attacks run with their default parameters (per-attack parameter overrides
  are a backend feature planned in `advanced_options`).
- Methods flagged **TF** need `pip install "the-a-files[ai]"`; methods flagged
  **long input** (ECHO, DSSS) fail per-row on short files by design.
- Uploaded datasets and experiment history are in-memory per API process.
