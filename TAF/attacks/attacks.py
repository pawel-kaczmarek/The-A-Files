from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.fft import fftfreq
from scipy.signal import butter, sosfilt

from TAF.models.WavFile import WavFile


@dataclass
class CorruptedWavFile(WavFile):

    def __init__(self, wav_file: WavFile):
        self.samples = wav_file.samples
        self.samplerate = wav_file.samplerate
        self.path = wav_file.path

    def low_pass_filter(self, order: int = 16, cutoff_freq: float = 2000) -> CorruptedWavFile:
        sos = butter(order, cutoff_freq, fs=self.samplerate, output='sos', btype='low')
        self.samples = sosfilt(sos, self.samples)
        return self

    def additive_noise(self, std: float = 0.001) -> CorruptedWavFile:
        self.samples = np.random.normal(0, std, self.samples.shape[0]) + self.samples
        return self

    def frequency_filter(self, cutoff_frequency=2000) -> CorruptedWavFile:
        W = np.fft.fftshift(fftfreq(len(self.samples), d=1 / self.samplerate))
        fft = np.fft.fftshift(np.fft.fft(self.samples))
        filtered_fft = fft.copy()
        filtered_fft[(np.abs(W) == cutoff_frequency)] = 0
        self.samples = np.fft.ifft(np.fft.ifftshift(filtered_fft))
        return self

    def flip_random_samples(self, samples_to_flip=200) -> CorruptedWavFile:
        to_flip = np.random.choice(len(self.samples), samples_to_flip, replace=False)
        data = self.samples.copy()
        data[to_flip] = -data[to_flip]
        self.samples = data
        return self

    def cut_random_samples(self, samples_to_cut=200) -> CorruptedWavFile:
        to_flip = np.random.choice(len(self.samples), samples_to_cut, replace=False)
        data = self.samples.copy()
        data[to_flip] = 0
        self.samples = data
        return self

    def resample(self, target_samplerate: int = 27500):
        self.samples = librosa.resample(
            self.samples, orig_sr=self.samplerate, target_sr=target_samplerate, scale=True
        )
        return self

    def amplitude_scaling(self, scale: float = 1.1) -> CorruptedWavFile:
        self.samples = np.multiply(self.samples, scale)
        return self

    def pitch_shift(self, n_steps: int = 4, bins_per_octave: int = 12) -> CorruptedWavFile:
        self.samples = librosa.effects.pitch_shift(
            self.samples, sr=self.samplerate, n_steps=n_steps, bins_per_octave=bins_per_octave
        )
        return self

    def time_stretch(self, rate: float = 2.0) -> CorruptedWavFile:
        self.samples = librosa.effects.time_stretch(self.samples, rate=rate)
        return self
