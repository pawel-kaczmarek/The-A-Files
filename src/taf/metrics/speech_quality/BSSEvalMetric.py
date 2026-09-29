from numbers import Number

import numpy as np

from taf.models.Metric import Metric
from taf.models.card import MetricCard, Reference, Text


class BSSEvalMetric(Metric):

    card = MetricCard(
        title="BSS Eval v4",
        abbreviation="BSSEval",
        category="speech_quality",
        scale="dB",
        domain="audio",
        references=(Reference("Stöter et al.", 2018, doi="10.5281/zenodo.3376621"),),
        summary=Text(
            en="Reports several distortion measures originally designed for source-separation evaluation.",
            pl="Zwraca kilka miar zniekształceń opracowanych do oceny separacji źródeł.",
        ),
        details=Text(
            en=(
                "Uses museval BSS Eval v4 to decompose estimation errors into target-related, "
                "interference and artefact terms. Returns SDR, ISR, SIR and SAR in dB, plus a source "
                "permutation index. Higher is better for the four ratios; perm identifies source "
                "assignment and is not a quality score. Some components can be degenerate for "
                "single-source comparisons."
            ),
            pl=(
                "Używa museval BSS Eval v4 do rozłożenia błędów estymacji na składniki związane z "
                "celem, interferencją i artefaktami. Zwraca SDR, ISR, SIR i SAR w dB oraz indeks "
                "permutacji źródeł. Więcej jest lepiej dla czterech ilorazów; perm określa przypisanie "
                "źródła i nie jest oceną jakości. Przy porównaniu jednego źródła część składników może "
                "być zdegenerowana."
            ),
        ),
    )

    higher_is_better = True
    components = ("sdr", "isr", "sir", "sar", "perm")
    # The permutation index identifies sources; it is not a score.
    component_directions = (True, True, True, True, None)

    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        try:
            from museval import metrics  # https://github.com/sigsep/sigsep-mus-eval
        except (ImportError, RuntimeError) as exc:
            raise ImportError(
                "BSS_EVAL requires museval and external ffmpeg/ffprobe binaries."
            ) from exc

        result = metrics.bss_eval(reference_sources=samples_original,  # shape: [nsrc, nsample, nchannels]
                                  estimated_sources=samples_processed)
        values = [item[0][0] for item in result]
        return np.asarray(values) # (sdr, isr, sir, sar, perm)

    def name(self) -> str:
        return "BSS_EVAL version 4."
