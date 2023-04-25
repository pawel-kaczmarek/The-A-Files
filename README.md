# The A-Files

```
  _______ _                                   ______ _ _           
 |__   __| |                 /\              |  ____(_) |          
    | |  | |__   ___        /  \     ______  | |__   _| | ___  ___ 
    | |  | '_ \ / _ \      / /\ \   |______| |  __| | | |/ _ \/ __|
    | |  | | | |  __/     / ____ \           | |    | | |  __/\__ \
    |_|  |_| |_|\___|    /_/    \_\          |_|    |_|_|\___||___/
                                                                    
```

Implementations of audio watermarking methods and speech quality metrics in different domains.

<font size="1">
Powered by some great [GitHub repositories](#links)
</font>

## Steganography algorithms

List of implemented methods:

|                         Class name | Description [Reference]                                                   |
|-----------------------------------:|---------------------------------------------------------------------------|
|                 ```LsbMethod.py``` | Standard LSB coding [[1]](#articles)                                      |
|                ```EchoMethod.py``` | Echo Hiding technique with single echo kernel [[1]](#articles)            |
|         ```PhaseCodingMethod.py``` | Phase coding technique [[1]](#articles)                                   |
|         ```DctDeltaLsbMethod.py``` | DCT Delta LSB [[1]](#articles)                                            |
|              ```DwtLsbMethod.py``` | DWT LSB based [[1]](#articles)                                            |
|               ```DctB1Method.py``` | First band of DCT coefficients (DCT-b1) [[2]](#articles)                  |
| ```PatchworkMultilayerMethod.py``` | Patchwork-Based multilayer [[3]](#articles)                               |
|           ```NormSpaceMethod.py``` | Norm space method [[4]](#articles)                                        |
|                ```FsvcMethod.py``` | Frequency Singular Value Coefficient Modification (FSVC) [[5]](#articles) |
|                ```DsssMethod.py``` | Direct Sequence Spread Spectrum technique [[6]](#articles)                |

Each method extend abstract class  ```SteganographyMethod```

```python
from abc import abstractmethod, ABC
from typing import List
import numpy as np


class SteganographyMethod(ABC):

    @abstractmethod
    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        ...

    @abstractmethod
    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        ...

    @abstractmethod
    def type(self) -> str:
        ...
```

## Metrics

#### Speech Reverberation

- Bark spectral distortion (BSD) [[7]](#articles)
- Speech-to-reverberation modulation energy ratio (SRMR) [[10]](#articles)

#### Speech Intelligibility

- Coherence and speech intelligibility index (CSII) [[7]](#articles)
- Normalized-covariance measure (NCM) [[7]](#articles)
- Short-time objective intelligibility (STOI) [[9]](#articles)

#### Speech Quality

- Signal - to - Noise Ratio (SNR) [[12]](#articles)
- Mel-cepstral distance measure for objective speech quality assessment [[11]](#articles)
- Segmental Signal-to-Noise Ratio (SNRseg) [[7]](#articles)
- Frequency-weighted Segmental SNR (fwSNRseg) [[7]](#articles)
- Cepstrum Distance Objective Speech Quality Measure (CD) [[7]](#articles)
- Log - likelihood Ratio (LLR) [[7]](#articles)
- Weighted Spectral Slope (WSS) [[7]](#articles)
- Perceptual Evaluation of Speech Quality (PESQ) [[8]](#articles)

Each metric extend abstract class  ```Metric```

```python
from abc import ABC, abstractmethod
from numbers import Number

import numpy as np


class Metric(ABC):

    @abstractmethod
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float,
                  overlap: float) -> Number | np.ndarray:
        ...

    @abstractmethod
    def name(self) -> str:
        ...
```


## Diagram

<img src="./Documentation/diagram.svg" alt="The A-Files diagram">

## References

#### Articles

[1] Alsabhany, Ahmed A., Ahmed Hussain Ali, Farida Ridzuan, A. H. Azni, and Mohd Rosmadi Mokhtar. “Digital Audio
Steganography: Systematic Review, Classification, and Analysis of the Current State of the Art.” Computer Science Review
38 (2020): 100316. https://doi.org/10.1016/j.cosrev.2020.100316 \
[2] Hu, Hwai Tsu, and Ling Yuan Hsu. “Robust, Transparent and High-Capacity Audio Watermarking in DCT Domain.” Signal
Processing 109 (2015): 226–35. https://doi.org/10.1016/j.sigpro.2014.11.011 \
[3] Natgunanathan, Iynkaran, Yong Xiang, Guang Hua, Gleb Beliakov, and John Yearwood. “Patchwork-Based Multilayer Audio
Watermarking.” IEEE/ACM Transactions on Audio Speech and Language Processing 25, no. 11 (2017):
2176–87. https://doi.org/10.1109/TASLP.2017.2749001 \
[4] Saadi, Slami, Ahmed Merrad, and Ali Benziane. “Novel Secured Scheme for Blind Audio/Speech Norm-Space Watermarking
by Arnold Algorithm.” Signal Processing 154 (2019): 74–86. https://doi.org/10.1016/j.sigpro.2018.08.011 \
[5] Zhao, Juan, Tianrui Zong, Yong Xiang, Longxiang Gao, Wanlei Zhou, and Gleb Beliakov. “Desynchronization Attacks
Resilient Watermarking Method Based on Frequency Singular Value Coefficient Modification.” IEEE/ACM Transactions on
Audio Speech and Language Processing 29 (2021): 2282–95. https://doi.org/10.1109/TASLP.2021.3092555 \
[6] Nugraha, Rizky M. “Implementation of Direct Sequence Spread Spectrum Steganography on Audio Data.” Proceedings of
the 2011 International Conference on Electrical Engineering and Informatics, ICEEI 2011, no. July (
2011). https://doi.org/10.1109/ICEEI.2011.6021662 \
[7] Philipos C. Loizou. “Speech Enhancement. Theory and Practice, Second Edition” CRC Press (
2013). https://doi.org/10.1201/b14529 \
[8] Miao Wang, Christoph Boeddeker, Rafael G. Dantas and ananda seelan. “PESQ (Perceptual Evaluation of Speech Quality)
Wrapper for Python Users” Zenodo 2022. https://doi.org/10.5281/zenodo.6549559 \
[9] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time Objective Intelligibility Measure for Time-Frequency
Weighted Noisy Speech', ICASSP 2010, Texas, Dallas. https://doi.org/10.1109/ICASSP.2010.5495701 \
[10] Tiago H. Falk, Chenxi Zheng, and Way-Yip Chan. A Non-Intrusive Quality and Intelligibility Measure of Reverberant
and Dereverberated Speech, IEEE Trans Audio Speech Lang Process, Vol. 18, No. 7, pp. 1766-1774,
Sept.2010. https://doi.org/10.1109/TASL.2010.2052247 \
[11] R. Kubichek, "Mel-cepstral distance measure for objective speech quality assessment," Proceedings of IEEE Pacific
Rim Conference on Communications Computers and Signal Processing, Victoria, BC, Canada, 1993, pp. 125-128
vol.1, https://doi.org/10.1109/PACRIM.1993.407206. \
[12] https://en.wikipedia.org/wiki/Signal-to-noise_ratio

#### Links

[1] https://github.com/kosta-pmf/audio-watermarking \
[2] https://github.com/ktekeli/audio-steganography-algorithms \
[3] https://github.com/schmiph2/pysepm \
[4] https://github.com/ludlows/PESQ \
[5] https://github.com/mpariente/pystoi \
[6] https://github.com/jfsantos/SRMRpy \
[7] https://github.com/jasminsternkopf/mel_cepstral_distance

### Licence

The A-Files is an open source software under GPLv3 license.

### Authors

- Paweł Kaczmarek ([@pawelkaczmarek12](https://github.com/pawelkaczmarek12)) - Military University of Technology,
  Faculty of Electronics
- Zbigniew Piotrowski - Military University of Technology, Faculty of Electronics
