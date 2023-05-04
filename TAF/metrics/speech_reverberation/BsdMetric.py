from numbers import Number

import numpy as np
from scipy.signal import stft, lfilter
from scipy.signal.windows import hann

from TAF.models.Metric import Metric


class BsdMetric(Metric):
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        pre_emphasis_coeff = 0.95
        b = np.array([1])
        a = np.array([1, pre_emphasis_coeff])
        samples_original = lfilter(b, a, samples_original)
        samples_processed = lfilter(b, a, samples_processed)

        winlength = round(frame_len * fs)  # window length in samples
        skiprate = int(np.floor((1 - overlap) * frame_len * fs))  # window skip in samples
        max_freq = fs / 2  # maximum bandwidth
        n_fft = 2 ** np.ceil(np.log2(2 * winlength))
        n_fftby2 = int(n_fft / 2)
        num_frames = len(samples_original) / skiprate - (winlength / skiprate)  # number of frames

        hannWin = hann(winlength)  # 0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
        f, t, Zxx = stft(samples_original[0:int(num_frames) * skiprate + int(winlength - skiprate)], fs=fs,
                         window=hannWin,
                         nperseg=winlength, noverlap=winlength - skiprate, nfft=n_fft, detrend=False,
                         return_onesided=True, boundary=None, padded=False)
        clean_power_spec = np.square(np.sum(hannWin) * np.abs(Zxx))
        f, t, Zxx = stft(samples_processed[0:int(num_frames) * skiprate + int(winlength - skiprate)], fs=fs,
                         window=hannWin, nperseg=winlength, noverlap=winlength - skiprate, nfft=n_fft, detrend=False,
                         return_onesided=True, boundary=None, padded=False)
        enh_power_spec = np.square(np.sum(hannWin) * np.abs(Zxx))

        bark_filt = self._barks(fs, n_fft, n_barks=32)
        clean_power_spec_bark = np.dot(bark_filt, clean_power_spec)
        enh_power_spec_bark = np.dot(bark_filt, enh_power_spec)

        clean_power_spec_bark_2 = np.square(clean_power_spec_bark)
        diff_power_spec_2 = np.square(clean_power_spec_bark - enh_power_spec_bark)

        bsd = np.mean(np.sum(diff_power_spec_2, axis=0) / np.sum(clean_power_spec_bark_2, axis=0))
        return bsd

    def name(self) -> str:
        return "Bark spectral distortion (BSD)"

    @staticmethod
    def _barks(fs, n_fft, n_barks=128, fmin=0.0, fmax=None, norm='area', dtype=np.float32):
        if fmax is None:
            fmax = float(fs) / 2

        # Initialize the weights
        n_barks = int(n_barks)
        weights = np.zeros((n_barks, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = np.linspace(0, float(fs) / 2, int(1 + n_fft // 2), endpoint=True)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        bark_f = BsdMetric._bark_frequencies(n_barks + 2, fmin=fmin, fmax=fmax)

        fdiff = np.diff(bark_f)
        ramps = np.subtract.outer(bark_f, fftfreqs)

        for i in range(n_barks):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        if norm in (1, 'area'):
            weightsPerBand = np.sum(weights, 1);
            for i in range(weights.shape[0]):
                weights[i, :] = weights[i, :] / weightsPerBand[i]
        return weights

    @staticmethod
    def _bark_frequencies(n_barks=128, fmin=0.0, fmax=11025.0):
        # 'Center freqs' of bark bands - uniformly spaced between limits
        min_bark = BsdMetric._hz_to_bark(fmin)
        max_bark = BsdMetric._hz_to_bark(fmax)

        barks = np.linspace(min_bark, max_bark, n_barks)

        return BsdMetric._bark_to_hz(barks)

    @staticmethod
    def _bark_to_hz(barks):
        barks = barks.copy()
        barks = np.asanyarray([barks])
        barks[barks < 2] = (barks[barks < 2] - 0.3) / 0.85
        barks[barks > 20.1] = (barks[barks > 20.1] + 4.422) / 1.22
        freqs_hz = 1960 * (barks + 0.53) / (26.28 - barks)
        return np.squeeze(freqs_hz)

    @staticmethod
    def _hz_to_bark(freqs_hz):
        freqs_hz = np.asanyarray([freqs_hz])
        barks = (26.81 * freqs_hz) / (1960 + freqs_hz) - 0.53
        barks[barks < 2] = barks[barks < 2] + 0.15 * (2 - barks[barks < 2])
        barks[barks > 20.1] = barks[barks > 20.1] + 0.22 * (barks[barks > 20.1] - 20.1)
        return np.squeeze(barks)
