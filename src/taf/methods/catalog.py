"""Descriptive metadata of the packaged methods.

The factory knows how to build a method; this module says what it is: the
family of embedding technique, whether it was designed for covert
communication (steganography) or for robust marking (watermarking), the
publication it implements, and which constructor parameter controls the
embedding strength - the knob a trade-off curve sweeps.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any

from taf.models.types import MethodType

#: Families of embedding techniques, in the order they are usually presented.
FAMILIES = (
    "lsb",
    "transform",
    "spread_spectrum",
    "echo",
    "phase",
    "quantization",
    "statistical",
    "adaptive",
    "learned",
    "neural",
)

#: Constructor arguments that are not experimental parameters.
_NOT_PARAMETERS = {"self", "sr", "args", "kwargs"}
#: Parameters that act as a secret key rather than as a tuning knob.
KEY_PARAMETERS = {"key", "seed", "hhat_seed"}


@dataclass(frozen=True)
class MethodMetadata:
    family: str
    purpose: str  # "steganography" or "watermarking"
    reference: str
    year: int
    doi: str | None = None
    #: Constructor parameter that scales the embedding strength, if any.
    strength_parameter: str | None = None


METHOD_METADATA: dict[MethodType, MethodMetadata] = {
    MethodType.LSB_METHOD: MethodMetadata("lsb", "steganography", "Alsabhany et al.", 2020, "10.1016/j.cosrev.2020.100316"),
    MethodType.ECHO_METHOD: MethodMetadata("echo", "steganography", "Alsabhany et al.", 2020, "10.1016/j.cosrev.2020.100316", "alpha"),
    MethodType.PHASE_CODING_METHOD: MethodMetadata("phase", "steganography", "Alsabhany et al.", 2020, "10.1016/j.cosrev.2020.100316"),
    MethodType.IMPROVED_PHASE_CODING_METHOD: MethodMetadata("phase", "steganography", "Yang", 2024, "10.48550/arXiv.2408.13277"),
    MethodType.DCT_DELTA_LSB_METHOD: MethodMetadata("transform", "steganography", "Alsabhany et al.", 2020, "10.1016/j.cosrev.2020.100316", "delta_value"),
    MethodType.DWT_LSB_METHOD: MethodMetadata("transform", "steganography", "Alsabhany et al.", 2020, "10.1016/j.cosrev.2020.100316", "step_scale"),
    MethodType.DCT_B1_METHOD: MethodMetadata("transform", "watermarking", "Hu & Hsu", 2015, "10.1016/j.sigpro.2014.11.011"),
    MethodType.PATCHWORK_MULTILAYER_METHOD: MethodMetadata("statistical", "watermarking", "Natgunanathan et al.", 2017, "10.1109/TASLP.2017.2749001"),
    MethodType.NORM_SPACE_METHOD: MethodMetadata("transform", "watermarking", "Saadi et al.", 2019, "10.1016/j.sigpro.2018.08.011", "delta"),
    MethodType.FSVC_METHOD: MethodMetadata("transform", "watermarking", "Zhao et al.", 2021, "10.1109/TASLP.2021.3092555", "delta"),
    MethodType.DSSS_METHOD: MethodMetadata("spread_spectrum", "steganography", "Nugraha", 2011, "10.1109/ICEEI.2011.6021662", "alpha"),
    MethodType.BLIND_SVD_METHOD: MethodMetadata("transform", "watermarking", "Dhar & Shimamura", 2015, "10.1016/j.jisa.2014.10.007", "quantization_coefficient"),
    MethodType.PRIME_FACTOR_INTERPOLATE: MethodMetadata("lsb", "steganography", "Adhiyaksa et al.", 2022, "10.1109/ISMODE53584.2022.9743066"),
    MethodType.LWT_METHOD: MethodMetadata("transform", "watermarking", "Mushtaq et al.", 2024, "10.1109/ICRITO61523.2024.10522195", "threshold"),
    MethodType.FBSMethod: MethodMetadata("lsb", "steganography", "Wang & Wang", 2025, "10.1016/j.compeleceng.2024.109247"),
    MethodType.FGAS_METHOD: MethodMetadata("neural", "steganography", "Yan et al.", 2025, "10.48550/arXiv.2505.22266", "epsilon"),
    MethodType.AAC_STC_METHOD: MethodMetadata("adaptive", "steganography", "Luo et al.", 2017, "10.1007/978-3-319-64185-0_14"),
    MethodType.WIRELESS_DWT_LSB_METHOD: MethodMetadata("transform", "steganography", "Hamdi et al.", 2025, "10.3390/jsan14060106"),
    MethodType.LEARNABLE_EMBEDDING_GA_METHOD: MethodMetadata("learned", "watermarking", "Nayeem et al.", 2026, "10.1016/j.dsp.2026.106372", "embedding_strength"),
    MethodType.QIM_METHOD: MethodMetadata("quantization", "watermarking", "Chen & Wornell", 2001, "10.1109/18.923725", "step_scale"),
    MethodType.IMPROVED_SPREAD_SPECTRUM_METHOD: MethodMetadata("spread_spectrum", "watermarking", "Malvar & Florencio", 2003, "10.1109/TSP.2003.809385", "strength"),
    MethodType.BACKWARD_FORWARD_ECHO_METHOD: MethodMetadata("echo", "watermarking", "Kim & Choi", 2003, "10.1109/TCSVT.2003.815950", "alpha"),
    MethodType.TIME_SPREAD_ECHO_METHOD: MethodMetadata("echo", "watermarking", "Ko et al.", 2005, "10.1109/TMM.2005.843366", "alpha"),
    MethodType.HISTOGRAM_METHOD: MethodMetadata("statistical", "watermarking", "Xiang & Huang", 2007, "10.1109/TMM.2007.906580", "threshold"),
    MethodType.LOW_FREQUENCY_AMPLITUDE_METHOD: MethodMetadata("statistical", "watermarking", "Lie & Chang", 2006, "10.1109/TMM.2005.861292", "margin"),
    MethodType.AUDIOSEAL_METHOD: MethodMetadata("neural", "watermarking", "San Roman et al.", 2024, "10.48550/arXiv.2401.17264", "alpha"),
    MethodType.WAVMARK_METHOD: MethodMetadata("neural", "watermarking", "Chen et al.", 2023, "10.48550/arXiv.2308.12770"),
}


#: Short names used in the literature where the description carries none.
_ABBREVIATIONS = {
    "QIM_METHOD": "QIM",
    "PRIME_FACTOR_INTERPOLATE": "PFI",
    "ECHO_METHOD": "Echo",
    "BACKWARD_FORWARD_ECHO_METHOD": "BF-Echo",
    "TIME_SPREAD_ECHO_METHOD": "TS-Echo",
    "PHASE_CODING_METHOD": "Phase",
    "IMPROVED_PHASE_CODING_METHOD": "IPC",
    "LWT_METHOD": "LWT",
    "NORM_SPACE_METHOD": "Norm-space",
    "PATCHWORK_MULTILAYER_METHOD": "Patchwork-ML",
    "BLIND_SVD_METHOD": "Blind-SVD",
    "WIRELESS_DWT_LSB_METHOD": "W-DWT-LSB",
    "DWT_LSB_METHOD": "DWT-LSB",
    "DCT_DELTA_LSB_METHOD": "DCT-Delta-LSB",
    "HISTOGRAM_METHOD": "Histogram",
    "AAC_STC_METHOD": "AAC-STC",
    "LEARNABLE_EMBEDDING_GA_METHOD": "LE-GA",
    "AUDIOSEAL_METHOD": "AudioSeal",
    "WAVMARK_METHOD": "WavMark",
}


def method_abbreviation(name: str, description: str = "") -> str:
    """A short label for figures: the catalogue's, the acronym in the
    description's parentheses, or the registry name without its suffix."""
    if name in _ABBREVIATIONS:
        return _ABBREVIATIONS[name]
    match = re.search(r"\(([^()]{2,12})\)", description)
    if match:
        return match.group(1)
    for suffix in ("_METHOD", "Method"):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", "-")


def constructor_parameters(cls: type) -> list[dict[str, Any]]:
    """Tunable constructor parameters of a method class, with their defaults.

    The sampling rate is supplied by the engine and ``*args``/``**kwargs``
    are not parameters. Parameters without a default are reported with
    ``required``.
    """
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return []
    parameters: list[dict[str, Any]] = []
    for name, parameter in signature.parameters.items():
        if name in _NOT_PARAMETERS or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        default = None if parameter.default is inspect.Parameter.empty else parameter.default
        parameters.append(
            {
                "name": name,
                "default": default,
                "type": type(default).__name__ if default is not None else None,
                "required": parameter.default is inspect.Parameter.empty,
                "is_key": name in KEY_PARAMETERS,
            }
        )
    return parameters


__all__ = [
    "FAMILIES",
    "KEY_PARAMETERS",
    "METHOD_METADATA",
    "MethodMetadata",
    "constructor_parameters",
    "method_abbreviation",
]
