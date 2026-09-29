# syntax=docker/dockerfile:1
#
# The A-Files research platform API (taf-api).
#
#   docker build -t the-a-files-api .
#   docker build -t the-a-files-api --build-arg EXTRAS=platform,neural .
#
# EXTRAS selects the optional dependency groups installed next to the package
# (see pyproject.toml): "platform" is required for the API, "ai" adds
# TensorFlow (FgasMethod, MosNetMetric), "neural" adds AudioSeal and WavMark.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build

ARG EXTRAS=platform

# pesq ships only as a source distribution and needs a C compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /src
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install ".[${EXTRAS}]"


FROM python:${PYTHON_VERSION}-slim

LABEL org.opencontainers.image.source="https://github.com/pawel-kaczmarek/The-A-Files" \
      org.opencontainers.image.description="The A-Files research platform API: audio steganography methods, attacks and metrics" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"

# FFmpeg runs the real encoders behind the codec attacks; libsndfile backs
# soundfile for WAV, FLAC and OGG.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 taf \
    && mkdir -p /data \
    && chown taf:taf /data

COPY --from=build /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib \
    TAF_API_HOST=0.0.0.0 \
    TAF_API_PORT=8000 \
    TAF_DATA_DIR=/data \
    TAF_DATABASE_URL=postgresql+psycopg://taf:taf@db:5432/taf

USER taf
WORKDIR /home/taf
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]

CMD ["taf-api"]
