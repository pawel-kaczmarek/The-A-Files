# The A-Files — web research platform

Next.js 15 + React 19 + TypeScript + Tailwind client for the
[The A-Files](../README.md) research platform. It is a **thin client**: every
research decision (experimental designs, sweeps, seeding, attacks, metrics,
statistics, Pareto fronts, reports, corpus preparation) is made by the Python
package and exposed through its REST API. The frontend collects a protocol,
shows progress and renders results.

> This directory is **not** part of the PyPI distribution.

## Prerequisites

- Node.js ≥ 18.18
- PostgreSQL and the taf API (from the repository root):

```powershell
docker compose up -d db              # PostgreSQL 17 on localhost:5432 (taf/taf/taf)
pip install -e ".[platform]"
taf-api                              # http://127.0.0.1:8000, OpenAPI at /docs
```

The API applies database migrations when it starts. `TAF_DATABASE_URL`
overrides the connection string and `TAF_DATA_DIR` (default `~/.taf`) holds
prepared corpora and uploads.

## Development

```powershell
cd web
npm install
npm run dev    # http://localhost:3000
npm run build  # production build; npx tsc --noEmit for a type check
```

Set `NEXT_PUBLIC_TAF_API_URL` (see `.env.example`) if the API runs elsewhere.

## Routes

| Route | Purpose |
|---|---|
| `/dashboard` | Overview: counts, recent runs, database status |
| `/experiments` | Saved experiment protocols (versioned, archivable) |
| `/experiments/new` | Design gallery grouped by property (imperceptibility, robustness, capacity, security, multi-criteria), then the editor |
| `/experiments/[id]` | Protocol, version history and runs of one experiment |
| `/experiments/[id]/edit` | Editor; saving a changed protocol creates a new version |
| `/runs` | All runs with live status (Server-Sent Events) |
| `/runs/[id]` | Results, Statistics, Trials, Provenance and Report tabs (`?tab=`) |
| `/runs/[id]/trials/[rowId]` | Trial inspector: the trial is re-synthesised from its seed; cover, stego, attacked and residual (stego − cover) playback with spectrograms, decoded vs. embedded message |
| `/methods`, `/methods/[name]` | Method catalogue with family, purpose, reference and tunable parameters |
| `/attacks`, `/metrics` | Attack and metric catalogues (severity levels, directions, components) |
| `/datasets`, `/datasets/[id]` | Library, standard corpora (reproducible subsets), uploads, local directories, synthetic signals |
| `/methodology` | Protocol and statistics the platform applies |
| `/settings` | Language, theme, API and database status, data directory |

The editor steps are Protocol → Data → Methods → Conditions → Measures →
Design parameters → Review; `?design=<type>&step=<step>` opens a step directly.
The review step shows the execution plan returned by
`POST /api/experiments/preview` (trial count, factorial breakdown, warnings).

## Conventions

- **Language.** English by default; Polish is selectable (header switch,
  `?lang=pl`, stored in `localStorage` as `taf-locale`). `lib/i18n/pl.ts` is
  typed against the English dictionary, so a missing key fails the build.
- **Theme.** Light and dark, following the system until chosen.
- **Figures.** Categorical colours come from fixed slots (`--series-1…8`),
  every figure has a table view, intervals are 95% cluster-bootstrap intervals
  over files, and colour is never the only carrier of meaning. Methods are
  labelled with the catalogue's short names; the full name is shown on hover.
- **Motion.** A few effects adapted from [Magic UI](https://magicui.design) (MIT)
  live in `components/magicui/`, ported to Tailwind 3 and the platform's
  tokens: number ticker, blur fade, border beam (runs in progress), magic card
  (design gallery), animated beam (the evaluation-model diagram), shimmer
  button (primary actions), shine border, word rotate, dot pattern. Motion is
  informational or decorative, never the only carrier of a value, and every
  effect stops under `prefers-reduced-motion` (`MotionConfig reducedMotion="user"`
  plus `motion-safe:` utilities).
- **Types.** `lib/types.ts` mirrors the Pydantic schemas in
  `src/taf/api/schemas.py`; the backend validates authoritatively.
