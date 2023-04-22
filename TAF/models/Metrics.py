from dataclasses import dataclass
from typing import List

import numpy as np

from TAF.metrics.speech_intelligibility.CsiiMetric import CsiiMetric
from TAF.metrics.speech_intelligibility.NcmMetric import NcmMetric
from TAF.metrics.speech_intelligibility.StoiMetric import StoiMetric
from TAF.metrics.speech_quality.CepstrumDistanceMetric import CepstrumDistanceMetric
from TAF.metrics.speech_quality.FWSnrSegMetric import FWSnrSegMetric
from TAF.metrics.speech_quality.LlrMetric import LlrMetric
from TAF.metrics.speech_quality.MelCepstralDistanceMetric import MelCepstralDistanceMetric
from TAF.metrics.speech_quality.PesqMetric import PesqMetric
from TAF.metrics.speech_quality.SnrMetric import SnrMetric
from TAF.metrics.speech_quality.SnrSegMetric import SnrSegMetric
from TAF.metrics.speech_quality.WssMetric import WssMetric
from TAF.metrics.speech_reverberation.BsdMetric import BsdMetric
from TAF.metrics.speech_reverberation.SrmrMetric import SrmrMetric
from TAF.models.Metric import Metric

_speech_quality_measures: List[Metric] = [
    SnrMetric(),
    SnrSegMetric(),
    FWSnrSegMetric(),
    PesqMetric(),
    WssMetric(),
    LlrMetric(),
    CepstrumDistanceMetric(),
    MelCepstralDistanceMetric()
]

_speech_intelligibility_measures: List[Metric] = [
    CsiiMetric(),
    NcmMetric(),
    StoiMetric(),
]

_speech_dereverberation_measures: List[Metric] = [
    SrmrMetric(),
    BsdMetric()
]


@dataclass
class Metrics:
    samples_original: np.ndarray
    samples_processed: np.ndarray
    fs: int
    frame_len: float = 0.03
    overlap: float = 0.75

    def calculate_speech_quality(self):
        return [{metric.name(): metric.calculate(
            self.samples_original, self.samples_processed, self.fs, self.frame_len, self.overlap
        )} for metric in _speech_quality_measures]

    def calculate_speech_intelligibility(self):
        return [{metric.name(): metric.calculate(
            self.samples_original, self.samples_processed, self.fs, self.frame_len, self.overlap
        )} for metric in _speech_intelligibility_measures]

    def calculate_speech_reverberation(self):
        return [{metric.name(): metric.calculate(
            self.samples_original, self.samples_processed, self.fs, self.frame_len, self.overlap
        )} for metric in _speech_dereverberation_measures]

    def calculate_all(self):
        return self.calculate_speech_quality() \
            + self.calculate_speech_intelligibility() \
            + self.calculate_speech_reverberation()
