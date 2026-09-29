from numbers import Number

import numpy as np
from librosa.feature import melspectrogram
from mel_cepstral_distance import compare_mel_spectrograms

from taf.models.Metric import Metric
from taf.models.card import MetricCard, Reference, Text


# https://ieeexplore.ieee.org/document/407206
class MelCepstralDistanceMetric(Metric):

    card = MetricCard(
        title="Mel-cepstral distance",
        abbreviation="MCD",
        category="speech_quality",
        scale="dB",
        references=(Reference("Kubichek", 1993, doi="10.1109/PACRIM.1993.407206"),),
        summary=Text(
            en="Measures changes in speech spectral shape using a mel-scaled representation.",
            pl="Mierzy zmiany kształtu widma mowy w reprezentacji o skali melowej.",
        ),
        details=Text(
            en=(
                "Builds 20-band mel power spectrograms with a 1024-sample Hamming window and 256-sample "
                "hop, then passes them to the mel-cepstral-distance comparison library. Lower MCD means "
                "closer representations. Values depend on feature extraction and comparison settings, "
                "so scores from differently configured MCD implementations are not automatically "
                "interchangeable."
            ),
            pl=(
                "Buduje 20-pasmowe melowe spektrogramy mocy z oknem Hamminga 1024 próbki i krokiem 256, "
                "a następnie przekazuje je do biblioteki mel-cepstral-distance. Niższe MCD oznacza "
                "bliższe reprezentacje. Wyniki zależą od ekstrakcji cech i ustawień porównania, więc "
                "wartości z różnych konfiguracji MCD nie są automatycznie porównywalne."
            ),
        ),
    )

    higher_is_better = False

    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        hop_length: int = 256
        n_fft: int = 1024
        window: str = 'hamming'
        center: bool = False
        n_mels: int = 20
        htk: bool = True
        norm = None
        dtype: np.dtype = np.float64

        mel_spectrogram_original = melspectrogram(
            y=samples_original,
            sr=fs,
            hop_length=hop_length,
            n_fft=n_fft,
            window=window,
            center=center,
            S=None,
            pad_mode="constant",
            power=2.0,
            win_length=None,
            n_mels=n_mels,
            htk=htk,
            norm=norm,
            dtype=dtype,
            fmin=0.0,
            fmax=None,
        )

        mel_spectrogram_processed = melspectrogram(
            y=samples_processed,
            sr=fs,
            hop_length=hop_length,
            n_fft=n_fft,
            window=window,
            center=center,
            S=None,
            pad_mode="constant",
            power=2.0,
            win_length=None,
            n_mels=n_mels,
            htk=htk,
            norm=norm,
            dtype=dtype,
            fmin=0.0,
            fmax=None,
        )

        # compare_mel_spectrograms expects (frames, n_mels); librosa returns (n_mels, frames).
        mcd, _ = compare_mel_spectrograms(
            mel_spectrogram_original.T,
            mel_spectrogram_processed.T,
        )
        return np.array(mcd)

    def name(self) -> str:
        return "Mel-cepstral distance measure for objective speech quality assessment"
