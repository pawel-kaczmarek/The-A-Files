"""Descriptive metadata of the packaged metrics: scale, reference, intrusiveness.

The metric classes declare what their numbers mean for ranking
(``higher_is_better``, ``components``); this module adds what a reader needs
to interpret a value: its unit or scale, whether it compares the processed
signal with the original (intrusive) or rates it on its own, and the
publication that defines it.
"""

from __future__ import annotations

from dataclasses import dataclass

from taf.models.types import MetricType


@dataclass(frozen=True)
class MetricMetadata:
    abbreviation: str
    scale: str
    reference: str
    year: int
    intrusive: bool = True


METRIC_METADATA: dict[MetricType, MetricMetadata] = {
    MetricType.SNR_METRIC: MetricMetadata("SNR", "dB", "Loizou", 2013),
    MetricType.SNR_SEG_METRIC: MetricMetadata("SNRseg", "dB", "Loizou", 2013),
    MetricType.FWSNR_SEG_METRIC: MetricMetadata("fwSNRseg", "dB", "Hu & Loizou", 2008),
    MetricType.PESQ_METRIC: MetricMetadata("PESQ", "MOS-LQO 1–4.64", "ITU-T P.862 / Wang et al.", 2022),
    MetricType.WSS_METRIC: MetricMetadata("WSS", "distance", "Hu & Loizou", 2008),
    MetricType.LLR_METRIC: MetricMetadata("LLR", "distance", "Hu & Loizou", 2008),
    MetricType.CEPSTRUM_DISTANCE_METRIC: MetricMetadata("CD", "dB", "Loizou", 2013),
    MetricType.MEL_CEPSTRAL_DISTANCE_METRIC: MetricMetadata("MCD", "dB", "Kubichek", 1993),
    MetricType.CSII_METRIC: MetricMetadata("CSII", "0–1", "Loizou", 2013),
    MetricType.NCM_METRIC: MetricMetadata("NCM", "0–1", "Loizou", 2013),
    MetricType.STOI_METRIC: MetricMetadata("STOI", "0–1", "Taal et al.", 2010),
    MetricType.SRMR_METRIC: MetricMetadata("SRMR", "ratio", "Falk et al.", 2010, intrusive=False),
    MetricType.BSD_METRIC: MetricMetadata("BSD", "distance", "Loizou", 2013),
    MetricType.CBAK_METRIC: MetricMetadata("Cbak", "1–5", "Hu & Loizou", 2008),
    MetricType.CSIG_METRIC: MetricMetadata("Csig", "1–5", "Hu & Loizou", 2008),
    MetricType.COVL_METRIC: MetricMetadata("Covl", "1–5", "Hu & Loizou", 2008),
    MetricType.STGI_METRIC: MetricMetadata("STGI", "0–1", "Edraki et al.", 2021),
    MetricType.WSTMI_METRIC: MetricMetadata("wSTMI", "index", "Edraki et al.", 2021),
    MetricType.SISDR_METRIC: MetricMetadata("SI-SDR", "dB", "Le Roux et al.", 2019),
    MetricType.BSS_EVAL_METRIC: MetricMetadata("BSSEval", "dB", "Stöter et al.", 2018),
    MetricType.AI_MOSNET_METRIC: MetricMetadata("MOSNet", "MOS 1–5", "Lo et al.", 2019, intrusive=False),
}

__all__ = ["METRIC_METADATA", "MetricMetadata"]
