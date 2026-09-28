# Installation and quick start

## Python package

Requires **Python 3.10?3.12**. The package is distributed on PyPI:

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

See [External dependencies](#external-dependencies) for system-level prerequisites (C++ build tools, FFmpeg).

## Usage

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

## External dependencies

* **PESQ**, when built from source on Windows, may require Microsoft Visual C++ 14.0 or later, available through the
  [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
* **FFmpeg** must be available on `PATH` for codec attacks and some format conversions: <https://ffmpeg.org/>.


## Optional models

AudioSeal and WavMark load released pretrained weights through their optional packages; first use may
require a download. FGAS and MOSNet require the TensorFlow extra.

ViSQOL Audio requires the official Google bindings and `libsvm_nu_svr_model.txt`, installed separately
following the [official instructions](https://github.com/google/visqol#python-api-usage). The TAF adapter
uses audio mode and resamples metric inputs to 48 kHz; this does not restore missing bandwidth. Missing
bindings or model files are reported as metric errors. Do not pool audio-mode and speech-mode scores.

For browser-based experiments, follow the [UI setup](ui.md#start-locally).
