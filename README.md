# The A-Files

```
  _______ _                                   ______ _ _           
 |__   __| |                 /\              |  ____(_) |          
    | |  | |__   ___        /  \     ______  | |__   _| | ___  ___ 
    | |  | '_ \ / _ \      / /\ \   |______| |  __| | | |/ _ \/ __|
    | |  | | | |  __/     / ____ \           | |    | | |  __/\__ \
    |_|  |_| |_|\___|    /_/    \_\          |_|    |_|_|\___||___/
                                                                    
```

> ***The A-Files is a powerful audio steganography software that allows users to embed secret data within an
audio signal, measure speech quality metrics, and test the audio signal's robustness against different types of attacks.
<br><br>With The A-Files, users can ensure that their sensitive information remains private and protected.***

## Table of contents

* [About](#about)
* [Steganography algorithms](#steganography-algorithms)
* [Metrics](#metrics)
    * [Speech Reverberation](#speech-reverberation)
    * [Speech Intelligibility](#speech-intelligibility)
    * [Speech Quality](#speech-quality)
* [Attacks](#attacks)
* [Diagram](#diagram)
* [References](#references)
* [Licence](#licence)
* [Authors](#authors)

## About

The A-Files project contains:

* Audio watermarking methods
* Speech quality metrics
* Audio attacks

<img src="./Documentation/functions.svg" alt="The A-Files functions"> 

###### Loading

The A-Files is an audio steganography software that allows users to hide secret data within an audio signal.
To do this, users can load both the audio file and the data they want to hide into the software. The A-Files supports
several types
of audio files, such as WAV or FLAC. Once the audio file and the secret data are loaded, the software can embed the data
within the audio signal.

###### Capacity

The A-Files offers several techniques for embedding secret data within an audio signal, such as LSB (Least Significant
Bit) insertion, phase coding, and echo hiding. Each technique has its own strengths and weaknesses, and users can choose
the technique that best fits their application's requirements.

###### Transparency

The A-Files includes tools for measuring speech quality metrics, such as PESQ (Perceptual Evaluation of Speech
Quality), SNR (Signal-to-Noise Ratio) and STOI (Short-time objective intelligibility) metric. By measuring these
metrics, users can ensure that the audio signal has not been degraded during the process of embedding secret data.

###### Robustness

The A-Files provides tools for testing the audio signal's robustness against different types of attacks. Attackers may
attempt to remove the hidden data or alter the audio signal to render the hidden data useless. The A-Files includes
tools for testing the audio signal's robustness against various types of attacks, such as noising, frequency cutting,
and filtering. By testing the audio signal's robustness, users can determine the effectiveness of the audio
steganography
technique and make any necessary adjustments to improve the technique's strength and resilience.

*Powered by some great [GitHub repositories](#links)*

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

[main.py](TAF%2Fmain.py)
[Metric.py](TAF%2Fmetrics%2Fspeech_quality%2Fcomposite%2FCsigMetric.py)
[Metric.py](TAF%2Fmetrics%2Fspeech_quality%2Fcomposite%2FCovlMetric.py)
[Metric.py](TAF%2Fmetrics%2Fspeech_quality%2Fcomposite%2FCbakMetric.py)

- Signal - to - Noise Ratio (SNR) [[12]](#articles)
- Mel-cepstral distance measure for objective speech quality assessment [[11]](#articles)
- Segmental Signal-to-Noise Ratio (SNRseg) [[7]](#articles)
- Frequency-weighted Segmental SNR (fwSNRseg) [[7]](#articles)
- Cepstrum Distance Objective Speech Quality Measure (CD) [[7]](#articles)
- Log - likelihood Ratio (LLR) [[7]](#articles)
- Weighted Spectral Slope (WSS) [[7]](#articles)
- Perceptual Evaluation of Speech Quality (PESQ) [[8]](#articles)
- Speech Enhancement Metrics (Csig, Covl, Cbak, Composite) [[13]](#articles)
- Weighted Spectro-Temporal Modulation Index (wSTMI) [[14]](#articles)
- Spectro-Temporal Glimpsing Index (STGI) [[15]](#articles)

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
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        ...

    @abstractmethod
    def name(self) -> str:
        ...
```

## Attacks

List of attack on audio samples:

* Low pass filter
* Additive noise
* Frequency filter
* Flip random samples
* Cut random samples
* Resample (downsampling, upsampling)
* Amplitude scaling
* Pitch shift
* Time stretch

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
[11] R. Kubichek, "Mel-cepstral distance measure for objective speech quality assessment", Proceedings of IEEE Pacific
Rim Conference on Communications Computers and Signal Processing, Victoria, BC, Canada, 1993, pp. 125-128
vol.1, https://doi.org/10.1109/PACRIM.1993.407206. \
[12] https://en.wikipedia.org/wiki/Signal-to-noise_ratio
[13] Yi Hu and Philipos C. Loizou, "Evaluation of Objective Quality Measures for Speech Enhancement", IEEE TRANSACTIONS
ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 16, NO. 1, JANUARY 2008 229
[14] A. Edraki, W.-Y. Chan, J. Jensen, & D. Fogerty, “Speech Intelligibility Prediction Using Spectro-Temporal Modulation
Analysis,” IEEE/ACM Trans. Audio, Speech, & Language Processing, vol. 29, pp. 210-225, 2021.
[15] A. Edraki, W.-Y. Chan, J. Jensen, & D. Fogerty, “A Spectro-Temporal Glimpsing Index (STGI) for Speech
Intelligibility Prediction," Proc. Interspeech, 5 pages, Aug 2021.

#### Links

[1] https://github.com/kosta-pmf/audio-watermarking \
[2] https://github.com/ktekeli/audio-steganography-algorithms \
[3] https://github.com/schmiph2/pysepm \
[4] https://github.com/ludlows/PESQ \
[5] https://github.com/mpariente/pystoi \
[6] https://github.com/jfsantos/SRMRpy \
[7] https://github.com/jasminsternkopf/mel_cepstral_distance
[8] https://github.com/nglehuy/semetrics
[9] https://github.com/aminEdraki/py-intelligibility

### Licence

The A-Files is an open source software under GPLv3 license.

### Authors

- Paweł Kaczmarek ([@pawelkaczmarek12](https://github.com/pawelkaczmarek12)) - Military University of Technology,
  Faculty of Electronics
- Zbigniew Piotrowski - Military University of Technology, Faculty of Electronics
