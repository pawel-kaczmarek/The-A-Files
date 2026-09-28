# Research UI

The browser client collects experiment settings and displays results. The Python backend executes
methods, attacks, metrics and statistical analyses; PostgreSQL stores protocols and runs.
GitHub Pages serves this documentation only.

## Start locally

Use Python 3.10–3.12, Node.js compatible with the web package (minimum 18.18), npm and Docker Compose.
From a repository checkout:

```bash
docker compose up -d db
pip install -e ".[platform]"
taf-api
```

Leave the API running. In another terminal:

```bash
cd web
npm ci
npm run dev
```

Open **http://localhost:3000**. The API listens at **http://127.0.0.1:8000**, with OpenAPI at `/docs`.
If the API is elsewhere, set `NEXT_PUBLIC_TAF_API_URL` in the web environment before starting the client.
Database migrations run when the API starts. See [API configuration](platform.md).

## Create a protocol

1. Open **Datasets** and select bundled speech, prepare a catalogued corpus, upload audio, register a
   server-local directory or generate synthetic signals. Check the selected files and sample rate.
2. Open **Experiments → New** and choose a design: quality, robustness, capacity, detectability,
   a parameter curve, comparison, dataset benchmark or a custom research experiment.
3. Follow the editor: **Protocol → Data → Methods → Conditions → Measures → Design parameters → Review**.
   Record the research question, choose methods and parameters, set payloads, attacks and metrics,
   and specify repetitions and a seed. Payloads may be random bits, text, hexadecimal bytes or explicit bits.
4. Read the preview’s trial count, factorial breakdown and warnings. Short files, excessive payloads,
   missing optional dependencies and small datasets can invalidate the intended comparison.
5. Save the protocol and start a run. Editing a saved protocol creates a new version; each run retains
   the version and configuration it executed. **Runs** displays live progress and cancellation controls.

## Interpret a run

| View | What to inspect |
| --- | --- |
| Results | Recovery, metric values and failure counts; compare clean and attacked conditions. |
| Statistics | File-based uncertainty intervals, paired comparisons, Pareto fronts and filtered research comparisons. |
| Trials | Individual result rows and failures; open a trial to inspect its signals and payload. |
| Provenance | Seeds, input hashes, resolved parameters and software versions. |
| Report | Generated narrative and tables; download Markdown or LaTeX. |

The trial inspector re-synthesises a selected trial from its configuration and seed. It offers cover,
stego, attacked and residual audio, spectrograms and the embedded versus decoded message. Use it to
examine a concrete failure before drawing a general conclusion.

Download detailed and summary CSV files, the configuration JSON and the provenance manifest with reports.
Keep failed trials visible: missing BER is not zero BER. File-level bootstrap intervals quantify uncertainty
under the selected protocol; they do not make a small bundled subset representative of all speech.

## Catalogues and preferences

**Methods**, **Attacks** and **Metrics** explain available components and parameters.
**Literature** contains sourced paper observations with their experimental conditions, separately from
local results. **Methodology** describes the analysis. The header and **Settings** provide English/Polish
and light/dark preferences; settings also show API and database status.

For automated use, see [experiments](experiments.md) and the [REST API](platform.md).
