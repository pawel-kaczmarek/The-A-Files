"""Optional official Google ViSQOL audio-mode adapter; no surrogate scores."""

from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from taf.metrics.common.spectral import aligned_mono
from taf.models.Metric import Metric
from taf.models.card import MetricCard, Reference, Text


def model_path() -> Path:
    try:
        from visqol import visqol_lib_py
    except ImportError as error:
        raise ImportError("ViSQOL requires the official google/visqol Python bindings and bundled SVR model; see https://github.com/google/visqol#python-api-usage") from error
    path = Path(visqol_lib_py.__file__).parent / "model" / "libsvm_nu_svr_model.txt"
    if not path.is_file():
        raise ImportError(f"ViSQOL audio SVR model is missing: {path}")
    return path


class VisqolMetric(Metric):

    card = MetricCard(
        title="ViSQOL Audio",
        abbreviation="ViSQOL Audio",
        category="speech_quality",
        scale="MOS-LQO (audio SVR, maximum about 4.75)",
        domain="audio",
        references=(
            Reference("Chinen et al.", 2020, doi="10.48550/arXiv.2004.09584"),
            Reference("google/visqol", None, url="https://github.com/google/visqol"),
        ),
        requires=("visqol",),
        summary=Text(
            en=(
                "Estimates perceived audio quality from similarity between reference and processed "
                "spectrograms."
            ),
            pl=(
                "Szacuje postrzeganą jakość dźwięku z podobieństwa spektrogramów oryginału i sygnału "
                "przetworzonego."
            ),
        ),
        details=Text(
            en=(
                "Uses official Google ViSQOL bindings and the audio-mode SVR model, with both inputs "
                "polyphase-resampled to 48 kHz. Spectro-temporal similarity is mapped to MOS-LQO, whose "
                "audio-model maximum is about 4.75. Prefer active clips around 8–10 seconds. Requires "
                "optional bindings and model data; do not compare these scores directly with ViSQOL "
                "speech mode."
            ),
            pl=(
                "Używa oficjalnych wiązań Google ViSQOL i modelu SVR trybu audio; oba wejścia są "
                "przeliczane wielofazowo do 48 kHz. Podobieństwo czasowo-widmowe jest mapowane na "
                "MOS-LQO z maksimum modelu audio około 4,75. Preferowane są aktywne fragmenty około "
                "8–10 sekund. Wymaga opcjonalnych wiązań i modelu; wyników nie należy bezpośrednio "
                "porównywać z trybem speech."
            ),
        ),
    )

    higher_is_better = True

    def calculate(self, samples_original, samples_processed, fs, frame_len=0.03, overlap=0.75):
        path = model_path()
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2

        x, y = aligned_mono(samples_original, samples_processed, fs)
        if not np.any(x):
            raise ValueError("ViSQOL requires a non-silent reference.")
        # Audio mode always uses the full-band 48 kHz SVR model, including for
        # speech. This resampling changes metric inputs only, never the decoder.
        if fs != 48000:
            divisor = gcd(fs, 48000)
            x = resample_poly(x, 48000 // divisor, fs // divisor)
            y = resample_poly(y, 48000 // divisor, fs // divisor)
        config = visqol_config_pb2.VisqolConfig()
        config.audio.sample_rate = 48000
        config.options.use_speech_scoring = False
        config.options.svr_model_path = str(path)
        api = visqol_lib_py.VisqolApi()
        api.Create(config)
        return float(api.Measure(np.ascontiguousarray(x), np.ascontiguousarray(y)).moslqo)

    def name(self):
        return "ViSQOL Audio (MOS-LQO)"
