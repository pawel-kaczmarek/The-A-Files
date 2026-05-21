from typing import Dict, List

from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.types import MethodType
from taf.methods.BlindSvdMethod import BlindSvdMethod
from taf.methods.DctB1Method import DctB1Method
from taf.methods.DctDeltaLsbMethod import DctDeltaLsbMethod
from taf.methods.DsssMethod import DsssMethod
from taf.methods.DwtLsbMethod import DwtLsbMethod
from taf.methods.EchoMethod import EchoMethod
from taf.methods.ForegroundBackgroundSegmentationMethod import ForegroundBackgroundSegmentationMethod
from taf.methods.AacStcMethod import AacStcMethod
from taf.methods.FgasMethod import FgasMethod
from taf.methods.FsvcMethod import FsvcMethod
from taf.methods.ImprovedPhaseCodingMethod import ImprovedPhaseCodingMethod
from taf.methods.LsbMethod import LsbMethod
from taf.methods.LwtMethod import LwtMethod
from taf.methods.NormSpaceMethod import NormSpaceMethod
from taf.methods.PatchworkMultilayerMethod import PatchworkMultilayerMethod
from taf.methods.PhaseCodingMethod import PhaseCodingMethod
from taf.methods.PrimeFactorInterpolatedMethod import PrimeFactorInterpolatedMethod

class SteganographyMethodFactory:

    @staticmethod
    def get(sr: int, methodType: MethodType) -> SteganographyMethod:
        return SteganographyMethodFactory._all_methods(sr).get(methodType)

    @staticmethod
    def get_all(sr: int) -> List[SteganographyMethod]:
        return [value for _, value in SteganographyMethodFactory._all_methods(sr).items()]

    @staticmethod
    def _all_methods(sr: int) -> Dict[MethodType, SteganographyMethod]:
        methods: Dict[MethodType, SteganographyMethod] = {
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
            MethodType.FGAS_METHOD: FgasMethod(sr=sr),
            MethodType.AAC_STC_METHOD: AacStcMethod(sr=sr),
        }
        return methods
