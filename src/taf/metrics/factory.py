from typing import Dict, List

from taf.metrics.ai_based.mosnet.MosNetMetric import MosNetMetric
from taf.metrics.speech_intelligibility.CsiiMetric import CsiiMetric
from taf.metrics.speech_intelligibility.NcmMetric import NcmMetric
from taf.metrics.speech_intelligibility.StgiMetric import StgiMetric
from taf.metrics.speech_intelligibility.StoiMetric import StoiMetric
from taf.metrics.speech_intelligibility.WstmiMetric import WstmiMetric
from taf.metrics.speech_quality.BSSEvalMetric import BSSEvalMetric
from taf.metrics.speech_quality.CepstrumDistanceMetric import CepstrumDistanceMetric
from taf.metrics.speech_quality.FWSnrSegMetric import FWSnrSegMetric
from taf.metrics.speech_quality.LlrMetric import LlrMetric
from taf.metrics.speech_quality.MelCepstralDistanceMetric import MelCepstralDistanceMetric
from taf.metrics.speech_quality.PesqMetric import PesqMetric
from taf.metrics.speech_quality.SisdrMetric import SisdrMetric
from taf.metrics.speech_quality.SnrMetric import SnrMetric
from taf.metrics.speech_quality.SnrSegMetric import SnrSegMetric
from taf.metrics.speech_quality.WssMetric import WssMetric
from taf.metrics.speech_quality.composite.CbakMetric import CbakMetric
from taf.metrics.speech_quality.composite.CovlMetric import CovlMetric
from taf.metrics.speech_quality.composite.CsigMetric import CsigMetric
from taf.metrics.speech_reverberation.BsdMetric import BsdMetric
from taf.metrics.speech_reverberation.SrmrMetric import SrmrMetric
from taf.models.Metric import Metric
from taf.models.types import MetricType


class MetricFactory:

    @staticmethod
    def get(metricType: MetricType) -> Metric:
        return MetricFactory._all_methods().get(metricType)

    @staticmethod
    def get_all() -> List[Metric]:
        return [value for _, value in MetricFactory._all_methods().items()]

    @staticmethod
    def _all_methods() -> Dict[MetricType, Metric]:
        return {
            MetricType.SNR_METRIC: SnrMetric(),
            MetricType.SNR_SEG_METRIC: SnrSegMetric(),
            MetricType.FWSNR_SEG_METRIC: FWSnrSegMetric(),
            MetricType.PESQ_METRIC: PesqMetric(),
            MetricType.WSS_METRIC: WssMetric(),
            MetricType.LLR_METRIC: LlrMetric(),
            MetricType.CEPSTRUM_DISTANCE_METRIC: CepstrumDistanceMetric(),
            MetricType.MEL_CEPSTRAL_DISTANCE_METRIC: MelCepstralDistanceMetric(),
            MetricType.CSII_METRIC: CsiiMetric(),
            MetricType.NCM_METRIC: NcmMetric(),
            MetricType.STOI_METRIC: StoiMetric(),
            MetricType.SRMR_METRIC: SrmrMetric(),
            MetricType.BSD_METRIC: BsdMetric(),
            MetricType.CBAK_METRIC: CbakMetric(),
            MetricType.CSIG_METRIC: CsigMetric(),
            MetricType.COVL_METRIC: CovlMetric(),
            MetricType.STGI_METRIC: StgiMetric(),
            MetricType.WSTMI_METRIC: WstmiMetric(),
            MetricType.SISDR_METRIC: SisdrMetric(),
            MetricType.BSS_EVAL_METRIC: BSSEvalMetric(),
            MetricType.AI_MOSNET_METRIC: MosNetMetric()
        }
