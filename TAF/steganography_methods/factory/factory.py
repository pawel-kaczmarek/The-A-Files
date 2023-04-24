from typing import List

from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.steganography_methods.DctB1Method import DctB1Method
from TAF.steganography_methods.DctDeltaLsbMethod import DctDeltaLsbMethod
from TAF.steganography_methods.DsssMethod import DsssMethod
from TAF.steganography_methods.DwtLsbMethod import DwtLsbMethod
from TAF.steganography_methods.EchoMethod import EchoMethod
from TAF.steganography_methods.FsvcMethod import FsvcMethod
from TAF.steganography_methods.LsbMethod import LsbMethod
from TAF.steganography_methods.NormSpaceMethod import NormSpaceMethod
from TAF.steganography_methods.PatchworkMultilayerMethod import PatchworkMultilayerMethod
from TAF.steganography_methods.PhaseCodingMethod import PhaseCodingMethod


def get_all_methods(sr: int) -> List[SteganographyMethod]:
    return [
        NormSpaceMethod(sr=sr),
        DwtLsbMethod(),
        FsvcMethod(sr=sr),
        DctB1Method(sr=sr),
        DctDeltaLsbMethod(sr=sr),
        PatchworkMultilayerMethod(sr=sr),
        PhaseCodingMethod(),
        DsssMethod(),
        LsbMethod(),
        EchoMethod(),
    ]
