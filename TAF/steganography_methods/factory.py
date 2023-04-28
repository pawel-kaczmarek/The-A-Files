from typing import Dict, List

from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.models.types import MethodType
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


class SteganographyMethodFactory:

    @staticmethod
    def get(sr: int, methodType: MethodType) -> SteganographyMethod:
        return SteganographyMethodFactory._all_methods(sr).get(methodType)

    @staticmethod
    def get_all(sr: int) -> List[SteganographyMethod]:
        return [value for _, value in SteganographyMethodFactory._all_methods(sr).items()]

    @staticmethod
    def _all_methods(sr: int) -> Dict[MethodType, SteganographyMethod]:
        return {
            MethodType.PHASE_CODING_METHOD: PhaseCodingMethod(),
            MethodType.DSSS_METHOD: DsssMethod(),
            MethodType.LSB_METHOD: LsbMethod(),
            MethodType.ECHO_METHOD: EchoMethod(),
            MethodType.DWT_LSB_METHOD: DwtLsbMethod(),
            MethodType.FSVC_METHOD: FsvcMethod(sr=sr),
            MethodType.DCT_B1_METHOD: DctB1Method(sr=sr),
            MethodType.DCT_DELTA_LSB_METHOD: DctDeltaLsbMethod(sr=sr),
            MethodType.PATCHWORK_MULTILAYER_METHOD: PatchworkMultilayerMethod(sr=sr),
            MethodType.NORM_SPACE_METHOD: NormSpaceMethod(sr=sr),
        }
