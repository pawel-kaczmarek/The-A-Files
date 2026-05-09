import random
from typing import List

import numpy as np


def generate_sinus_waveform(sampling_rate: int = 16000, duration: int = 2, f: float = 1000) -> np.ndarray:
    return (np.sin(2 * np.pi * np.arange(sampling_rate * duration) * f / sampling_rate)).astype(np.float32)


def generate_noise(size: int, mu: int = 0, sigma: int = 0.1) -> np.ndarray:
    return np.random.normal(mu, sigma, size)


def generate_random_message(length: int = 20) -> List[int]:
    return [random.choice([0, 1]) for _ in range(length)]
