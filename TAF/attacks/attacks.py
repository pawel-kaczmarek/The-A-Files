import librosa
import numpy as np
from scipy.fft import fftfreq
from scipy.signal import butter, sosfilt


class Attacks:

    def __init__(self, data: np.ndarray, sr: int):
        self.data = data
        self.sr = sr

    def low_pass_filter(self, order: int = 16, cutoff_freq: float = 2000):
        sos = butter(order, cutoff_freq, fs=self.sr, output='sos', btype='low')
        return sosfilt(sos, self.data)

    def additive_noise(self, std: float = 0.001):
        return np.random.normal(0, std, self.data.shape[0]) + self.data

    def frequency_filter(self, cutoff_frequency=2000):
        W = np.fft.fftshift(fftfreq(len(self.data), d=1 / self.sr))
        fft = np.fft.fftshift(np.fft.fft(self.data))
        filtered_fft = fft.copy()
        filtered_fft[(np.abs(W) == cutoff_frequency)] = 0
        return np.fft.ifft(np.fft.ifftshift(filtered_fft))

    def flip_random_samples(self, samples_to_flip=200):
        to_flip = np.random.choice(len(self.data), samples_to_flip, replace=False)
        filpped_data = self.data.copy()
        filpped_data[to_flip] = -filpped_data[to_flip]
        return filpped_data

    def cut_random_samples(self, samples_to_cut=200):
        to_flip = np.random.choice(len(self.data), samples_to_cut, replace=False)
        filpped_data = self.data.copy()
        filpped_data[to_flip] = 0
        return filpped_data

    def resample(self, target_sr: int = 27500):
        return librosa.resample(self.data, orig_sr=self.sr, target_sr=target_sr, scale=True)

    def amplitude_scaling(self, scale: float = 1.1):
        return np.multiply(self.data, scale)

    def pitch_shift(self, n_steps: int = 4, bins_per_octave: int = 12):
        return librosa.effects.pitch_shift(self.data, sr=self.sr, n_steps=n_steps, bins_per_octave=bins_per_octave)

    def time_stretch(self, rate: float = 2.0):
        return librosa.effects.time_stretch(self.data, rate=rate)
