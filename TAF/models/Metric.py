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
