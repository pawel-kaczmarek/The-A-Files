from __future__ import annotations

import unittest
from importlib.metadata import version

import numpy as np

import taf
from taf.models.types import MethodType, MetricType


class PackageImportTests(unittest.TestCase):
    def test_taf_import(self) -> None:
        self.assertTrue(taf)
        self.assertEqual(version("the-a-files"), "1.0.0")

    def test_core_enum_imports(self) -> None:
        self.assertTrue(MethodType)
        self.assertTrue(MetricType)

    def test_steganography_factory_importable_without_optional_extras(self) -> None:
        from taf.methods.factory import SteganographyMethodFactory

        methods = SteganographyMethodFactory.get_all(sr=16000)
        self.assertGreater(len(methods), 0)

    def test_metric_factory_importable_without_optional_extras(self) -> None:
        from taf.metrics.factory import MetricFactory

        metrics = MetricFactory.get_all()
        self.assertGreater(len(metrics), 0)

    def test_lsb_encode_decode_roundtrip(self) -> None:
        from taf.methods.factory import SteganographyMethodFactory

        rng = np.random.default_rng(seed=0)
        audio = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(16000) / 16000)).astype(np.float32)
        message = [int(b) for b in rng.integers(0, 2, size=16)]

        method = SteganographyMethodFactory.get(16000, MethodType.LSB_METHOD)
        encoded = method.encode(audio.copy(), list(message))
        decoded = [int(b) for b in method.decode(encoded, len(message))]

        self.assertEqual(decoded, message)

    def test_snr_metric_runs_without_optional_extras(self) -> None:
        from taf.metrics.factory import MetricFactory

        rng = np.random.default_rng(seed=1)
        audio = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(16000) / 16000)).astype(np.float32)
        noisy = audio + rng.normal(0.0, 0.01, size=audio.shape).astype(np.float32)

        snr = MetricFactory.get(MetricType.SNR_METRIC)
        result = snr.calculate(audio, noisy, 16000)

        self.assertIsNotNone(result)
        self.assertGreater(float(result), 0.0)


if __name__ == "__main__":
    unittest.main()
