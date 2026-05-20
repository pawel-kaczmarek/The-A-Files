import unittest
from importlib.metadata import version

import taf
from taf.models.types import MethodType, MetricType


class PackageImportTests(unittest.TestCase):
    def test_taf_import(self) -> None:
        self.assertTrue(taf)
        self.assertEqual(version("the-a-files"), "0.1.0")

    def test_core_enum_imports(self) -> None:
        self.assertTrue(MethodType)
        self.assertTrue(MetricType)


if __name__ == "__main__":
    unittest.main()
