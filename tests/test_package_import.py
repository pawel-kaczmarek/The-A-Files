import unittest

import taf
from taf.models.types import MethodType, MetricType


class PackageImportTests(unittest.TestCase):
    def test_taf_import(self) -> None:
        self.assertTrue(taf.__version__)

    def test_core_enum_imports(self) -> None:
        self.assertTrue(MethodType)
        self.assertTrue(MetricType)


if __name__ == "__main__":
    unittest.main()
