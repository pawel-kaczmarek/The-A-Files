from typing import List

from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.stganography.DctB1Method import DctB1Method
from TAF.stganography.DctDeltaLsbMethod import DctDeltaLsbMethod
from TAF.stganography.DsssMethod import DsssMethod
from TAF.stganography.DwtLsbMethod import DwtLsbMethod
from TAF.stganography.EchoMethod import EchoMethod
from TAF.stganography.FsvcMethod import FsvcMethod
from TAF.stganography.LsbMethod import LsbMethod
from TAF.stganography.NormSpaceMethod import NormSpaceMethod
from TAF.stganography.PatchworkMultilayerMethod import PatchworkMultilayerMethod
from TAF.stganography.PhaseCodingMethod import PhaseCodingMethod


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
