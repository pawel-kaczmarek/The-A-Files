# API and datasets

## Platform services

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

The complete platform also runs in containers from a repository checkout:

```bash
docker compose up -d --build     # web client on :3000, API on :8000, PostgreSQL on :5432
```

The API image (`Dockerfile`) includes FFmpeg for the codec attacks and keeps `TAF_DATA_DIR` in the `taf-data` volume.
Build-time options: `TAF_EXTRAS` selects the optional dependency groups (default `platform`; e.g. `platform,neural`),
and `TAF_PUBLIC_API_URL` is the API address as seen from the browser (default `http://localhost:8000`), which the web
image inlines at build time; when it changes, set `TAF_API_CORS_ORIGINS` to the web client's origin. Folders to
register with `POST /api/datasets/local` must be mounted into the `api` container (see `docker-compose.yml`) and
referenced by their path inside it.

| Purpose | Endpoints |
| --- | --- |
| Catalogue | `GET /api/catalog/{methods,metrics,attacks,designs,presets,corpora,datasets}` |
| Experiments | `GET, POST /api/experiments`, `GET, PUT, DELETE /api/experiments/{id}`, `POST /api/experiments/{id}/{duplicate,archive}`, `POST /api/experiments/preview` |
| Runs | `POST /api/experiments/{id}/runs`, `GET /api/runs`, `GET /api/runs/{id}/{summary,rows,facets,manifest.json,config.json}`, `POST /api/runs/{id}/cancel` |
| Exports and reports | `GET /api/runs/{id}/{export.csv,export_summary.csv,report.tex,report.md}` |
| Trial inspector | `GET /api/runs/{id}/rows/{row}/inspect`, `GET /api/runs/{id}/rows/{row}/audio/{cover,stego,attacked,residual}.wav` |
| Progress (Server-Sent Events) | `GET /api/runs/events`, `GET /api/runs/{id}/events` |
| Datasets | `GET /api/datasets`, `POST /api/datasets/{prepare,local,upload,synthetic}`, `GET, DELETE /api/datasets/{id}` |

The web client in [`web/`](https://github.com/pawelkaczmarek12/the-a-files/blob/master/web/README.md) (Next.js, TypeScript; English and Polish; light and dark theme) is a thin
client of this API. It offers a design gallery grouped by property, a step-by-step protocol editor with an execution
plan, live runs, result figures with table views, critical-difference diagrams, the trial inspector and report
downloads. It is not part of the PyPI distribution:

```bash
cd web
npm install
npm run dev   # http://localhost:3000
```

## Evaluation corpora

`taf.corpora` catalogues standard corpora with licence, citation and DOI: speech (LibriSpeech, Mini LibriSpeech,
VCTK 0.92, TSP, LJSpeech, LibriTTS-R, EARS, TIMIT, Common Voice, NOIZEUS), synthetic speech (ASVspoof 5, MLAAD), music
(MUSDB18-HQ, GTZAN) and environmental sound (ESC-50). Corpora with an open download are prepared on the server as
reproducible subsets by a recorded rule: a seeded selection balanced across speakers, conversion to mono, polyphase
resampling to the target rate, optional excerpts, 16-bit FLAC, and a manifest with the SHA-256 digest of the archive and
of every file. Licensed corpora are registered from a local copy. Uploads, local directories and a set of synthetic test
signals (tones, sweep, white and pink noise, tone bursts, square wave, low-level noise) complete the library. The
packaged VCTK and LibriSpeech subsets remain available without download.
