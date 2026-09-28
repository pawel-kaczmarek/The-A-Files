import inspect
from typing import Any, Callable, Dict, List

from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.types import MethodType
from taf.methods.AudioSealMethod import AudioSealMethod
from taf.methods.BackwardForwardEchoMethod import BackwardForwardEchoMethod
from taf.methods.BlindSvdMethod import BlindSvdMethod
from taf.methods.DctB1Method import DctB1Method
from taf.methods.DctDeltaLsbMethod import DctDeltaLsbMethod
from taf.methods.DsssMethod import DsssMethod
from taf.methods.DwtLsbMethod import DwtLsbMethod
from taf.methods.EchoMethod import EchoMethod
from taf.methods.EmdMethod import EmdMethod
from taf.methods.ForegroundBackgroundSegmentationMethod import ForegroundBackgroundSegmentationMethod
from taf.methods.AacStcMethod import AacStcMethod
from taf.methods.FgasMethod import FgasMethod
from taf.methods.FsvcMethod import FsvcMethod
from taf.methods.HistogramMethod import HistogramMethod
from taf.methods.ImprovedPhaseCodingMethod import ImprovedPhaseCodingMethod
from taf.methods.ImprovedSpreadSpectrumMethod import ImprovedSpreadSpectrumMethod
from taf.methods.LearnableEmbeddingGaMethod import LearnableEmbeddingGaMethod
from taf.methods.LowFrequencyAmplitudeMethod import LowFrequencyAmplitudeMethod
from taf.methods.LsbMethod import LsbMethod
from taf.methods.LwtMethod import LwtMethod
from taf.methods.NormSpaceMethod import NormSpaceMethod
from taf.methods.PatchworkMultilayerMethod import PatchworkMultilayerMethod
from taf.methods.PhaseCodingMethod import PhaseCodingMethod
from taf.methods.QimMethod import QimMethod
from taf.methods.ReversiblePeeMethod import ReversiblePeeMethod
from taf.methods.PrimeFactorInterpolatedMethod import PrimeFactorInterpolatedMethod
from taf.methods.SyncDwtDctMethod import SyncDwtDctMethod
from taf.methods.TimeSpreadEchoMethod import TimeSpreadEchoMethod
from taf.methods.WavMarkMethod import WavMarkMethod
from taf.methods.WirelessDwtLsbMethod import WirelessDwtLsbMethod

#: Class of every packaged method.
BUILTIN_METHOD_CLASSES: Dict[MethodType, type] = {
    MethodType.BLIND_SVD_METHOD: BlindSvdMethod,
    MethodType.IMPROVED_PHASE_CODING_METHOD: ImprovedPhaseCodingMethod,
    MethodType.PHASE_CODING_METHOD: PhaseCodingMethod,
    MethodType.DSSS_METHOD: DsssMethod,
    MethodType.LSB_METHOD: LsbMethod,
    MethodType.ECHO_METHOD: EchoMethod,
    MethodType.DWT_LSB_METHOD: DwtLsbMethod,
    MethodType.FSVC_METHOD: FsvcMethod,
    MethodType.DCT_B1_METHOD: DctB1Method,
    MethodType.DCT_DELTA_LSB_METHOD: DctDeltaLsbMethod,
    MethodType.PATCHWORK_MULTILAYER_METHOD: PatchworkMultilayerMethod,
    MethodType.NORM_SPACE_METHOD: NormSpaceMethod,
    MethodType.PRIME_FACTOR_INTERPOLATE: PrimeFactorInterpolatedMethod,
    MethodType.LWT_METHOD: LwtMethod,
    MethodType.FBSMethod: ForegroundBackgroundSegmentationMethod,
    MethodType.FGAS_METHOD: FgasMethod,
    MethodType.AAC_STC_METHOD: AacStcMethod,
    MethodType.WIRELESS_DWT_LSB_METHOD: WirelessDwtLsbMethod,
    MethodType.LEARNABLE_EMBEDDING_GA_METHOD: LearnableEmbeddingGaMethod,
    MethodType.QIM_METHOD: QimMethod,
    MethodType.IMPROVED_SPREAD_SPECTRUM_METHOD: ImprovedSpreadSpectrumMethod,
    MethodType.BACKWARD_FORWARD_ECHO_METHOD: BackwardForwardEchoMethod,
    MethodType.TIME_SPREAD_ECHO_METHOD: TimeSpreadEchoMethod,
    MethodType.HISTOGRAM_METHOD: HistogramMethod,
    MethodType.LOW_FREQUENCY_AMPLITUDE_METHOD: LowFrequencyAmplitudeMethod,
    MethodType.AUDIOSEAL_METHOD: AudioSealMethod,
    MethodType.WAVMARK_METHOD: WavMarkMethod,
    MethodType.SYNC_DWT_DCT_METHOD: SyncDwtDctMethod,
    MethodType.EMD_METHOD: EmdMethod,
    MethodType.REVERSIBLE_PEE_METHOD: ReversiblePeeMethod,
}


def build_method(cls: Callable[..., SteganographyMethod], sr: int, **parameters: Any) -> SteganographyMethod:
    """Instantiate ``cls`` with ``parameters``, passing the sampling rate if it takes one."""
    try:
        accepts_rate = "sr" in inspect.signature(cls).parameters
    except (TypeError, ValueError):
        accepts_rate = False
    if accepts_rate:
        return cls(sr=sr, **parameters)
    return cls(**parameters)


#: Constructor of every packaged method, called with the sampling rate. Only
#: the requested method is instantiated; building all of them for each lookup
#: made every ``get()`` pay for 30 constructors.
BUILTIN_METHODS: Dict[MethodType, Callable[[int], SteganographyMethod]] = {
    method_type: (lambda sr, cls=cls: build_method(cls, sr))
    for method_type, cls in BUILTIN_METHOD_CLASSES.items()
}


class SteganographyMethodFactory:

    @staticmethod
    def get(sr: int, methodType: MethodType) -> SteganographyMethod:
        create = BUILTIN_METHODS.get(methodType)
        return create(sr) if create is not None else None

    @staticmethod
    def get_all(sr: int) -> List[SteganographyMethod]:
        return list(SteganographyMethodFactory._all_methods(sr).values())

    @staticmethod
    def _all_methods(sr: int) -> Dict[MethodType, SteganographyMethod]:
        return {method_type: create(sr) for method_type, create in BUILTIN_METHODS.items()}
