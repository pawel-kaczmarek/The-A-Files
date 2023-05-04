from numbers import Number

import numpy as np
from scipy.signal import stft

from TAF.models.Metric import Metric


class WssMetric(Metric):
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        Kmax = 20  # value suggested by Klatt, pg 1280
        Klocmax = 1  # value suggested by Klatt, pg 1280
        alpha = 0.95
        if samples_original.shape != samples_processed.shape:
            raise ValueError('The two signals do not match!')
        eps = np.finfo(np.float64).eps
        samples_original = samples_original.astype(np.float64) + eps
        samples_processed = samples_processed.astype(np.float64) + eps
        winlength = round(frame_len * fs)  # window length in samples
        skiprate = int(np.floor((1 - overlap) * frame_len * fs))  # window skip in samples
        max_freq = fs / 2  # maximum bandwidth
        num_crit = 25  # number of critical bands
        n_fft = 2 ** np.ceil(np.log2(2 * winlength))
        n_fftby2 = int(n_fft / 2)

        cent_freq = np.zeros((num_crit,))
        bandwidth = np.zeros((num_crit,))

        cent_freq[0] = 50.0000
        bandwidth[0] = 70.0000
        cent_freq[1] = 120.000
        bandwidth[1] = 70.0000
        cent_freq[2] = 190.000
        bandwidth[2] = 70.0000
        cent_freq[3] = 260.000
        bandwidth[3] = 70.0000
        cent_freq[4] = 330.000
        bandwidth[4] = 70.0000
        cent_freq[5] = 400.000
        bandwidth[5] = 70.0000
        cent_freq[6] = 470.000
        bandwidth[6] = 70.0000
        cent_freq[7] = 540.000
        bandwidth[7] = 77.3724
        cent_freq[8] = 617.372
        bandwidth[8] = 86.0056
        cent_freq[9] = 703.378
        bandwidth[9] = 95.3398
        cent_freq[10] = 798.717
        bandwidth[10] = 105.411
        cent_freq[11] = 904.128
        bandwidth[11] = 116.256
        cent_freq[12] = 1020.38
        bandwidth[12] = 127.914
        cent_freq[13] = 1148.30
        bandwidth[13] = 140.423
        cent_freq[14] = 1288.72
        bandwidth[14] = 153.823
        cent_freq[15] = 1442.54
        bandwidth[15] = 168.154
        cent_freq[16] = 1610.70
        bandwidth[16] = 183.457
        cent_freq[17] = 1794.16
        bandwidth[17] = 199.776
        cent_freq[18] = 1993.93
        bandwidth[18] = 217.153
        cent_freq[19] = 2211.08
        bandwidth[19] = 235.631
        cent_freq[20] = 2446.71
        bandwidth[20] = 255.255
        cent_freq[21] = 2701.97
        bandwidth[21] = 276.072
        cent_freq[22] = 2978.04
        bandwidth[22] = 298.126
        cent_freq[23] = 3276.17
        bandwidth[23] = 321.465
        cent_freq[24] = 3597.63
        bandwidth[24] = 346.136

        W = np.array(
            [0.003, 0.003, 0.003, 0.007, 0.010, 0.016, 0.016, 0.017, 0.017, 0.022, 0.027, 0.028, 0.030, 0.032, 0.034,
             0.035,
             0.037, 0.036, 0.036, 0.033, 0.030, 0.029, 0.027, 0.026,
             0.026])

        bw_min = bandwidth[0]
        min_factor = np.exp(-30.0 / (2.0 * 2.303))  # % -30 dB point of filter

        all_f0 = np.zeros((num_crit,))
        crit_filter = np.zeros((num_crit, int(n_fftby2)))
        j = np.arange(0, n_fftby2)

        for i in range(num_crit):
            f0 = (cent_freq[i] / max_freq) * (n_fftby2)
            all_f0[i] = np.floor(f0)
            bw = (bandwidth[i] / max_freq) * (n_fftby2)
            norm_factor = np.log(bw_min) - np.log(bandwidth[i])
            crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + norm_factor)
            crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)

        num_frames = len(samples_original) / skiprate - (winlength / skiprate)  # number of frames
        start = 1  # starting sample

        hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
        scale = np.sqrt(1.0 / hannWin.sum() ** 2)

        f, t, Zxx = stft(samples_original[0:int(num_frames) * skiprate + int(winlength - skiprate)], fs=fs,
                         window=hannWin,
                         nperseg=winlength, noverlap=winlength - skiprate, nfft=n_fft, detrend=False,
                         return_onesided=True,
                         boundary=None, padded=False)
        clean_spec = np.power(np.abs(Zxx) / scale, 2)
        clean_spec = clean_spec[:-1, :]

        f, t, Zxx = stft(samples_processed[0:int(num_frames) * skiprate + int(winlength - skiprate)], fs=fs,
                         window=hannWin,
                         nperseg=winlength, noverlap=winlength - skiprate, nfft=n_fft, detrend=False,
                         return_onesided=True,
                         boundary=None, padded=False)
        proc_spec = np.power(np.abs(Zxx) / scale, 2)
        proc_spec = proc_spec[:-1, :]

        clean_energy = (crit_filter.dot(clean_spec))
        log_clean_energy = 10 * np.log10(clean_energy)
        log_clean_energy[log_clean_energy < -100] = -100
        proc_energy = (crit_filter.dot(proc_spec))
        log_proc_energy = 10 * np.log10(proc_energy)
        log_proc_energy[log_proc_energy < -100] = -100

        log_clean_energy_slope = np.diff(log_clean_energy, axis=0)
        log_proc_energy_slope = np.diff(log_proc_energy, axis=0)

        dBMax_clean = np.max(log_clean_energy, axis=0)
        dBMax_processed = np.max(log_proc_energy, axis=0)

        numFrames = log_clean_energy_slope.shape[-1]

        clean_loc_peaks = np.zeros_like(log_clean_energy_slope)
        proc_loc_peaks = np.zeros_like(log_proc_energy_slope)
        for ii in range(numFrames):
            clean_loc_peaks[:, ii] = self.find_loc_peaks(log_clean_energy_slope[:, ii], log_clean_energy[:, ii])
            proc_loc_peaks[:, ii] = self.find_loc_peaks(log_proc_energy_slope[:, ii], log_proc_energy[:, ii])

        Wmax_clean = Kmax / (Kmax + dBMax_clean - log_clean_energy[:-1, :])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peaks - log_clean_energy[:-1, :])
        W_clean = Wmax_clean * Wlocmax_clean

        Wmax_proc = Kmax / (Kmax + dBMax_processed - log_proc_energy[:-1])
        Wlocmax_proc = Klocmax / (Klocmax + proc_loc_peaks - log_proc_energy[:-1, :])
        W_proc = Wmax_proc * Wlocmax_proc

        W = (W_clean + W_proc) / 2.0

        distortion = np.sum(W * (log_clean_energy_slope - log_proc_energy_slope) ** 2, axis=0)
        distortion = distortion / np.sum(W, axis=0)
        distortion = np.sort(distortion)
        distortion = distortion[:int(round(len(distortion) * alpha))]
        return np.mean(distortion)

    def name(self) -> str:
        return "Weighted Spectral Slope (WSS)"

    # @jit
    @staticmethod
    def find_loc_peaks(slope, energy):
        num_crit = len(energy)

        loc_peaks = np.zeros_like(slope)

        for ii in range(len(slope)):
            n = ii
            if slope[ii] > 0:
                while (n < num_crit - 1) and (slope[n] > 0):
                    n = n + 1
                loc_peaks[ii] = energy[n - 1]
            else:
                while (n >= 0) and (slope[n] <= 0):
                    n = n - 1
                loc_peaks[ii] = energy[n + 1]

        return loc_peaks
