from typing import List

from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.steganography.DctB1Method import DctB1Method
from TAF.steganography.DctDeltaLsbMethod import DctDeltaLsbMethod
from TAF.steganography.DsssMethod import DsssMethod
from TAF.steganography.DwtLsbMethod import DwtLsbMethod
from TAF.steganography.EchoMethod import EchoMethod
from TAF.steganography.FsvcMethod import FsvcMethod
from TAF.steganography.LsbMethod import LsbMethod
from TAF.steganography.NormSpaceMethod import NormSpaceMethod
from TAF.steganography.PatchworkMultilayerMethod import PatchworkMultilayerMethod
from TAF.steganography.PhaseCodingMethod import PhaseCodingMethod


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
