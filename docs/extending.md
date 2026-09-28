# Extending the toolkit

## Plugins

Methods, metrics and attacks from other distributions are discovered through entry points, without modifying this
package. A registered component appears in the catalogue, in the API and in experiments under its entry-point name, and
is subject to the same method contract:

```toml
# pyproject.toml of the distribution that provides the components
[project.entry-points."taf.methods"]
MY_METHOD = "my_package.method:MyMethod"      # callable taking the sampling rate

[project.entry-points."taf.metrics"]
MY_METRIC = "my_package.metric:MyMetric"      # callable taking no arguments

[project.entry-points."taf.attacks"]
my_attack = "my_package.attack:MyAttack"      # Attack subclass
```

Packaged names take precedence, so a published result that names a packaged method always refers to the packaged
implementation (`taf.plugins`). An automated test enforces the package layering: the building blocks (`methods`,
`metrics`, `attacks`, `models`, `audio`, `steganalysis`) never import the evaluation engine, the experiment layer or the
HTTP layer. The engine in turn never imports the HTTP layer.


## Method interface

All methods implement the abstract interface `SteganographyMethod`:

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


## Metric interface

All metrics implement the abstract interface `Metric`:

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

Packaged additions must be registered in the corresponding factory and enum. See the [method contract](protocol.md#method-contract).
