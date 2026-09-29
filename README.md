<p align="center"><img src="https://raw.githubusercontent.com/pawel-kaczmarek/The-A-Files/master/docs/assets/banner.png" alt="The A-Files — audio steganography & watermarking toolkit" width="100%"></p>

# The A-Files

Research toolkit for evaluating **audio steganography and watermarking**: payload recovery,
signal quality, robustness, capacity and empirical detectability under a shared experimental protocol.
Includes a Python API, declarative experiments and a browser-based research interface.

**[Documentation](https://pawel-kaczmarek.github.io/The-A-Files/)** · [UI guide](https://pawel-kaczmarek.github.io/The-A-Files/ui/) · [References](https://pawel-kaczmarek.github.io/The-A-Files/references/)

<p align="center"><img src="https://raw.githubusercontent.com/pawel-kaczmarek/The-A-Files/master/docs/functions.svg" alt="The A-Files evaluation architecture: embed, attack, decode with a fresh decoder, then BER, transparency, detectability and per-file statistics" width="100%"></p>

## Quick start

Python **3.10–3.12**:

```bash
pip install the-a-files
taf-eval direct-no-metrics
```

Bundled VCTK and LibriSpeech subsets support initial experiments. FFmpeg is needed for codec attacks;
PESQ may require C++ build tools. See [installation and optional models](https://pawel-kaczmarek.github.io/The-A-Files/installation/).

<!-- catalogue:readme-methods -->
## Methods · 30

- **Sample-domain LSB:** FBS-LSB, LSB, PFI.
- **Transform domain:** Blind-SVD, DCT-b1, DCT-Delta-LSB, DWT-LSB, EMD, FSVC, LWT, Norm-space, Sync-DWT-DCT, W-DWT-LSB.
- **Spread spectrum:** DSSS, ISS.
- **Echo hiding:** BF-Echo, Echo, TS-Echo.
- **Phase coding:** IPC, Phase.
- **Quantisation (QIM):** QIM.
- **Statistical / patchwork:** Histogram, LFAM, Patchwork-ML.
- **Adaptive coding:** AAC-STC.
- **Reversible (lossless):** PEE.
- **Learned embedding:** LE-GA.
- **Neural network:** AudioSeal, FGAS, WavMark.

[Mechanisms, parameters, implementation limits and paper references](https://pawel-kaczmarek.github.io/The-A-Files/methods/).
<!-- /catalogue:readme-methods -->

<!-- catalogue:readme-attacks -->
## Attacks · 26

- **Additive noise:** `awgn`, `impulse_noise`, `pink_noise`.
- **Lossy codecs:** `codec`, with `aac`, `mp3`, `opus`, `vorbis` shortcuts.
- **Filtering:** `band_pass`, `high_pass`, `low_pass`, `notch`, `smoothing`.
- **Resampling & clock:** `clock_drift`, `resample`.
- **Quantisation:** `bit_depth`.
- **Amplitude & dynamics:** `clipping`, `compression_dynamic`, `gain`.
- **Temporal & desynchronisation:** `crop`, `dropout`, `pitch_shift`, `sample_jitter`, `speed`, `time_shift`, `time_stretch`, `zero_padding`.
- **Acoustic channel:** `acoustic_channel`, `echo`, `reverb`.
- **Pipelines:** `streaming_upload`, `voice_call`, `broadcast`, `over_the_air`, `desync_attack`.

[Individual descriptions, parameters, severity levels and scientific context](https://pawel-kaczmarek.github.io/The-A-Files/attacks/).
<!-- /catalogue:readme-attacks -->

<!-- catalogue:readme-metrics -->
## Metrics · 25

- **Speech quality:** BSSEval, Cbak, CD, Covl, Csig, fwSNRseg, LLR, LSD, MCD, MRSC, PESQ, SI-SDR, SNR, SNRseg, ViSQOL Audio, WSS.
- **Speech intelligibility:** CSII, eSTOI, NCM, STGI, STOI, wSTMI.
- **Reverberation:** BSD, SRMR.
- **Learned (AI-based):** MOSNet.

[Definitions, score directions and references](https://pawel-kaczmarek.github.io/The-A-Files/metrics/).
<!-- /catalogue:readme-metrics -->

BER, exact-message recovery, payload rate and processing time are reported separately.

## Adding a method, metric or attack

Write the class with its **card** (title, summary and details in English and Polish, references,
requirements) and register it in one place per kind. The API, the research UI, these lists and the
documentation pages are derived from it, and contract tests pick the component up automatically:

```bash
python -m taf.catalogue_docs     # regenerate the catalogue sections of the docs and this README
python -m pytest tests/test_catalogue.py tests/test_attack_contract.py tests/test_methods_roundtrip.py
```

[Step-by-step guide with templates](https://pawel-kaczmarek.github.io/The-A-Files/extending/).
Third-party packages can add components through entry points without modifying this repository.

## Research UI

The whole platform (PostgreSQL, API and web client) runs in Docker from the published images; only
`docker-compose.yml` is needed:

```bash
curl -O https://raw.githubusercontent.com/pawel-kaczmarek/The-A-Files/master/docker-compose.yml
docker compose up -d --no-build
```

The web client is at **http://localhost:3000** and the API at **http://localhost:8000** (OpenAPI at `/docs`).
Prepared corpora and uploads live in the `taf-data` volume. `TAF_IMAGE_TAG` selects a release (default `latest`;
`edge` follows `master`). In a repository checkout, `docker compose up -d --build` builds the images locally;
`TAF_EXTRAS=platform,neural` (or `platform,ai`) then adds the optional neural baselines or TensorFlow to the API.

For development, start only the database and run the API and web client locally:

```bash
docker compose up -d db
pip install -e ".[platform]"
taf-api
```

In a second terminal:

```bash
cd web
npm ci
npm run dev
```

Open **http://localhost:3000**. Create an experiment, select data, methods, payloads, attacks and metrics,
review the execution plan, then start a run. Results include statistics, individual trials with audio playback,
provenance and CSV/Markdown/LaTeX exports. English and Polish are available.
[Step-by-step UI guide](https://pawel-kaczmarek.github.io/The-A-Files/ui/).

Quality is measured separately for embedding and attack damage. Published results and local measurements
remain distinct; implementation adaptations are documented in the [method catalogue](https://pawel-kaczmarek.github.io/The-A-Files/methods/).
GitHub Pages hosts the documentation; the research UI runs with the Python API and PostgreSQL.

## Licence and authors

[GPL-3.0-or-later](https://github.com/pawel-kaczmarek/The-A-Files/blob/master/LICENSE). Paweł Kaczmarek and Zbigniew Piotrowski — Military University of Technology,
Faculty of Electronics. [Scientific background and bibliography](https://pawel-kaczmarek.github.io/The-A-Files/references/).
