"""Standard corpora: catalogue, download and reproducible subset preparation."""

from taf.corpora.catalog import CORPORA, CORPORA_BY_ID, DOMAINS, Corpus, Download, get_corpus
from taf.corpora.prepare import SubsetRule, download, prepare_subset, scan_directory, sha256_of
from taf.corpora.synthetic import synthetic_signals, write_synthetic_set

__all__ = [
    "CORPORA",
    "CORPORA_BY_ID",
    "DOMAINS",
    "Corpus",
    "Download",
    "SubsetRule",
    "download",
    "get_corpus",
    "prepare_subset",
    "scan_directory",
    "sha256_of",
    "synthetic_signals",
    "write_synthetic_set",
]
