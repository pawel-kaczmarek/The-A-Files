# Steganalysis

Transparency metrics quantify the perceptual cost of embedding and attacks quantify the survival of the payload, but
neither establishes whether the *presence* of a payload can be inferred — the criterion that distinguishes
steganography from watermarking. `taf.steganalysis` estimates this empirically.

Cover signals are segmented into windows; each window is paired with its stego counterpart, and the pairs are split
into disjoint training and test sets so that no cover contributes to both. Residual-Markov and log-spectral features
are extracted, and an ensemble of Fisher linear discriminants trained on random feature subspaces
[39](references.md#ref-39) is fitted to the training set. The procedure reports test accuracy, false-positive and
false-negative rates and the out-of-bag error of the ensemble.

```python
from taf.steganalysis import measure_detectability
from taf.methods.factory import SteganographyMethodFactory
from taf.models.types import MethodType

result = measure_detectability(
    SteganographyMethodFactory.get(16000, MethodType.LSB_METHOD),
    covers,                 # mono waveforms, segmented into windows internally
    message_length=20,
)
print(result.accuracy, result.false_positive_rate, result.undetectable)
```

A test accuracy of `0.5` corresponds to chance level and `1.0` to perfect detection. The result carries a 95% Wilson
interval of the accuracy and a one-sided exact binomial test against chance; `significantly_detectable` is `True` when
p < 0.05. The older `undetectable` flag (accuracy ≤ 0.55) is a heuristic on the point estimate, and with a small test
set it can disagree with the test. The result is a lower bound on detectability: a negative outcome does not exclude
detection by stronger features or classifiers. The `detectability` experiment type runs this analysis for every
selected method and payload length. Its window length and test fraction are set through `advanced_options`.
