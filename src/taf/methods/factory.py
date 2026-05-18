from typing import Dict, List

from models.SteganographyMethod import SteganographyMethod
from models.types import MethodType
from methods.BlindSvdMethod import BlindSvdMethod
from methods.DctB1Method import DctB1Method
from methods.DctDeltaLsbMethod import DctDeltaLsbMethod
from methods.DsssMethod import DsssMethod
from methods.DwtLsbMethod import DwtLsbMethod
from methods.EchoMethod import EchoMethod
from methods.ForegroundBackgroundSegmentationMethod import ForegroundBackgroundSegmentationMethod
from methods.FsvcMethod import FsvcMethod
from methods.ImprovedPhaseCodingMethod import ImprovedPhaseCodingMethod
from methods.LsbMethod import LsbMethod
from methods.LwtMethod import LwtMethod
from methods.NormSpaceMethod import NormSpaceMethod
from methods.PatchworkMultilayerMethod import PatchworkMultilayerMethod
from methods.PhaseCodingMethod import PhaseCodingMethod
from methods.PrimeFactorInterpolatedMethod import PrimeFactorInterpolatedMethod


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
            MethodType.BLIND_SVD_METHOD: BlindSvdMethod(),
            MethodType.IMPROVED_PHASE_CODING_METHOD: ImprovedPhaseCodingMethod(),
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
            MethodType.PRIME_FACTOR_INTERPOLATE: PrimeFactorInterpolatedMethod(),
            MethodType.LWT_METHOD: LwtMethod(),
            MethodType.FBSMethod: ForegroundBackgroundSegmentationMethod(sr=sr),
        }
