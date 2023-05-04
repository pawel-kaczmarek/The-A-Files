from numbers import Number

import numpy as np
from librosa.feature import melspectrogram
from mel_cepstral_distance import get_metrics_mels

from TAF.models.Metric import Metric


# https://ieeexplore.ieee.org/document/407206
class MelCepstralDistanceMetric(Metric):
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
        norm: str = None
        dtype: np.dtype = np.float64
        n_mfcc: int = 16
        use_dtw: bool = True

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
            # librosa.filters.mel arguments:
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
            # librosa.filters.mel arguments:
            n_mels=n_mels,
            htk=htk,
            norm=norm,
            dtype=dtype,
            fmin=0.0,
            fmax=None,
        )
        return np.array(get_metrics_mels(mel_spectrogram_original, mel_spectrogram_processed))

    def name(self) -> str:
        return "Mel-cepstral distance measure for objective speech quality assessment"
